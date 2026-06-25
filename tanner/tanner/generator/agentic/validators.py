from __future__ import annotations

import base64
import json
import re
from typing import Iterable
from urllib.parse import unquote, urlsplit

from tanner.generator.agentic.models import (
    ArtifactDraft,
    GeneratedArtifact,
    GeneratedBundle,
    GenerationRequest,
    GeneratorRuntimeConfig,
    PlannedArtifact,
    PlannedAssetFetch,
    ResourcePlan,
)


_LINK_RE = re.compile(r'(?:href|src|action)=["\']([^"\']+)["\']', re.I)
_CSS_URL_RE = re.compile(r"url\(\s*([^)]+?)\s*\)", re.I)
_JS_PATH_LITERAL_RE = re.compile(r"[\"'](/[^\"'\s?#]+(?:\?[^\"']*)?)[\"']")
_JS_EXTERNAL_URL_RE = re.compile(r"https?://[^\"'\s)]+", re.I)
_CONFIG_THEFT_SUPPORT_KINDS = {"config_text", "log_excerpt", "backup_manifest", "credential_bait"}
_INTERNAL_LANGUAGE_RE = re.compile(r"\b(fake|lure|attacker|attackers|honeypot)\b", re.I)
_BINARY_ASSET_EXTENSIONS = (
    ".ico",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
)
_LOGIN_PATH_RE = re.compile(
    r"(?:^|/)(?:wp-login|login|logon|signin|sign-in|auth|account|session|formlogin|form-login)(?:$|[/?._-])",
    re.IGNORECASE,
 )
_FORM_TAG_RE = re.compile(r"<form\b[^>]*>", re.IGNORECASE)
_FORM_METHOD_RE = re.compile(r"\bmethod\s*=\s*[\"\']?([^\"\'\s>]+)", re.IGNORECASE)
_FORM_ACTION_RE = re.compile(r"\baction\s*=\s*[\"\']([^\"\']+)", re.IGNORECASE)


class ValidationError(ValueError):
    pass


def normalize_path(path: str, index_page: str = "/index.html") -> str:
    if not isinstance(path, str) or not path.strip():
        return index_page

    normalized = urlsplit(path).path or "/"
    normalized = unquote(normalized)
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    normalized = re.sub(r"/+", "/", normalized)
    if normalized == "/":
        return index_page
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized or index_page


def ensure_generation_request(host: str | None, path: str, site_profile: dict | None) -> GenerationRequest:
    profile = site_profile.copy() if isinstance(site_profile, dict) else {}
    index_page = profile.get("index_page") if isinstance(profile.get("index_page"), str) else "/index.html"
    return GenerationRequest(
        host=host.strip() if isinstance(host, str) and host.strip() else None,
        requested_path=path,
        normalized_path=normalize_path(path, index_page=index_page),
        site_profile=profile,
        request_kind="seed" if profile.get("seed_request") else "runtime_miss",
        index_page=index_page,
    )


def infer_intent_family(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".env") or lowered in {"/.env", "/wp-config.php", "/config.php", "/settings.py"}:
        return "config_theft"
    if any(marker in lowered for marker in ["wp-admin", "wp-login", "wordpress"]):
        return "cms_probe"
    if any(lowered.endswith(ext) for ext in [".bak", ".backup", ".old", ".sql", ".zip", ".tar", ".gz"]):
        return "backup_probe"
    if any(marker in lowered for marker in ["/admin", "/login", "/dashboard", "/account"]):
        return "admin_portal"
    if any(marker in lowered for marker in ["/api", "/graphql", "/.git", "/server-status"]):
        return "framework_probe"
    return "generic_recon"


def _has_config_theft_support(artifacts, primary_path: str) -> bool:
    for artifact in artifacts:
        if getattr(artifact, "path", None) == primary_path:
            continue
        if getattr(artifact, "kind", None) in _CONFIG_THEFT_SUPPORT_KINDS:
            return True
    return False


def _is_form_handler_like_path(path: str) -> bool:
    lowered = path.lower()
    if lowered.startswith("/_flow/"):
        return False
    if lowered.endswith(_BINARY_ASSET_EXTENSIONS):
        return False
    return any(
        hint in lowered
        for hint in (
            "formlogin",
            "form-login",
            "login.cgi",
            "loginform",
            "login_submit",
            "submitlogin",
            "checklogin",
            "auth.cgi",
        )
    )


def _is_login_like_path(path: str) -> bool:
    lowered = path.lower()
    if lowered.startswith("/_flow/"):
        return False
    if lowered.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".otf")):
        return False
    return bool(_LOGIN_PATH_RE.search(lowered))


def _collect_login_post_targets(bundle: GeneratedBundle, request: GenerationRequest) -> set[str]:
    targets: set[str] = set()
    for artifact in bundle.artifacts:
        if artifact.kind != "html_page" or artifact.path.startswith("/_flow/"):
            continue
        body_text = artifact.body_bytes.decode("utf-8", errors="replace")
        for form_tag in _FORM_TAG_RE.findall(body_text):
            method_match = _FORM_METHOD_RE.search(form_tag)
            method = method_match.group(1).strip().upper() if method_match else "GET"
            if method != "POST":
                continue
            action_match = _FORM_ACTION_RE.search(form_tag)
            action = action_match.group(1).strip() if action_match else artifact.path
            if not action or _is_external_reference(action):
                continue
            normalized = normalize_path(action, index_page=request.index_page)
            targets.add(normalized)
    return targets


_FLOW_FAILURE_FEEDBACK_RE = re.compile(
    r"\b(invalid|incorrect|wrong|failed|failure|denied|unauthorized|try again|password is error|credential)\b",
    re.IGNORECASE,
)
_LOCKOUT_FEEDBACK_RE = re.compile(
    r"\b(too many|locked|lockout|wait 1 minute|wait one minute|try again later|temporarily blocked|too many invalid)\b",
    re.IGNORECASE,
)


def _artifact_has_failure_feedback(artifact: GeneratedArtifact | None) -> bool:
    if artifact is None:
        return False
    body_text = artifact.body_bytes.decode("utf-8", errors="replace")
    return bool(_FLOW_FAILURE_FEEDBACK_RE.search(body_text))


def _artifact_has_lockout_feedback(artifact: GeneratedArtifact | None) -> bool:
    if artifact is None:
        return False
    body_text = artifact.body_bytes.decode("utf-8", errors="replace")
    return bool(_LOCKOUT_FEEDBACK_RE.search(body_text))


def _validate_login_flow_rules(descriptor, login_targets: set[str], artifact_by_path: dict[str, GeneratedArtifact]) -> None:
    for target in sorted(login_targets):
        post_rules = [
            rule
            for rule in descriptor.rules
            if normalize_path(rule.match_path) == target
            and rule.condition is not None
            and (rule.condition.method or "").upper() == "POST"
        ]
        if not post_rules:
            raise ValidationError(
                "login-like POST target {} requires at least one POST flow rule".format(target)
            )
        artifact_post_rules = [rule for rule in post_rules if rule.response.artifact_path]
        if not artifact_post_rules:
            raise ValidationError(
                "form POST target {} requires at least one POST flow rule with artifact_path".format(target)
            )
        if not any(
            _artifact_has_failure_feedback(artifact_by_path.get(rule.response.artifact_path))
            for rule in artifact_post_rules
        ):
            raise ValidationError(
                "form POST target {} requires a POST artifact response with visible invalid-credential feedback".format(target)
            )
        lockout_rules = [
            rule
            for rule in artifact_post_rules
            if rule.condition is not None
            and rule.condition.lockout_active is True
            and rule.condition.lockout_window_seconds == 60
            and rule.condition.min_prior_post_count_to_path == 3
        ]
        if not lockout_rules:
            raise ValidationError(
                "form POST target {} requires a one-minute lockout artifact after three prior invalid attempts".format(target)
            )
        if not any(
            _artifact_has_lockout_feedback(artifact_by_path.get(rule.response.artifact_path))
            for rule in lockout_rules
        ):
            raise ValidationError(
                "form POST target {} requires a visible too-many-attempts lockout artifact".format(target)
            )
        distinct_outcomes = {
            (rule.response.artifact_path, rule.response.redirect_to, rule.response.status_code)
            for rule in post_rules
        }
        if len(distinct_outcomes) < 2:
            raise ValidationError(
                "form POST target {} requires at least two distinct POST flow outcomes".format(target)
            )

def _allowed_kinds_for_path(path: str) -> set[str] | None:
    lowered = path.lower()
    if lowered == "/robots.txt":
        return {"robots_txt"}
    if lowered == "/sitemap.xml":
        return {"sitemap_xml"}
    if lowered.endswith(".xml"):
        return {"xml_document"}
    if lowered.endswith(".json"):
        return {"json_document", "backup_manifest"}
    if lowered.endswith(".txt"):
        return {"plain_text", "config_text", "credential_bait", "log_excerpt", "backup_manifest"}
    if lowered.endswith(_BINARY_ASSET_EXTENSIONS):
        return {"binary_asset"}
    return None


def _binary_content_type_for_path(path: str) -> str | None:
    lowered = path.lower()
    if lowered.endswith(".ico"):
        return "image/x-icon"
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lowered.endswith(".gif"):
        return "image/gif"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".bmp"):
        return "image/bmp"
    if lowered.endswith(".svg"):
        return "image/svg+xml"
    if lowered.endswith(".woff"):
        return "font/woff"
    if lowered.endswith(".woff2"):
        return "font/woff2"
    if lowered.endswith(".ttf"):
        return "font/ttf"
    if lowered.endswith(".otf"):
        return "font/otf"
    return None


def _normalize_content_type(content_type: str | None) -> str | None:
    if not isinstance(content_type, str) or not content_type.strip():
        return None
    return content_type.split(";", 1)[0].strip().lower()


def _parse_content_type(content_type: str | None) -> tuple[str, str] | None:
    normalized = _normalize_content_type(content_type)
    if normalized is None or "/" not in normalized:
        return None
    top_level, subtype = normalized.split("/", 1)
    if not top_level or not subtype:
        return None
    return top_level, subtype


def _content_type_matches(actual: str | None, expected: str | None) -> bool:
    normalized_actual = _normalize_content_type(actual)
    normalized_expected = _normalize_content_type(expected)
    if normalized_expected is None:
        return True
    if normalized_actual == normalized_expected:
        return True

    parsed_actual = _parse_content_type(actual)
    parsed_expected = _parse_content_type(expected)
    if parsed_actual is None or parsed_expected is None:
        return False

    actual_top_level, actual_subtype = parsed_actual
    expected_top_level, expected_subtype = parsed_expected

    # Broad generic matching for data serialization formats: if both expected and actual
    # indicate the same structured syntax (e.g. json, xml, yaml), treat them as compatible.
    for generic_type in ("json", "xml", "yaml", "yml", "toml", "markdown", "md", "csv"):
        if generic_type in expected_subtype and generic_type in actual_subtype:
            return True

    # Text/plain artifacts (like config_text, plain_text, log_excerpt) can legitimately
    # be served as various structured text formats in real-world applications.
    if normalized_expected == "text/plain":
        if actual_top_level == "text":
            return True
        for generic_type in ("json", "xml", "yaml", "yml", "toml", "markdown", "md", "csv"):
            if generic_type in actual_subtype:
                return True

    # Otherwise, text artifacts are intentionally liberal: real appliances and frameworks
    # often label textual handler output as text/html, text/xml, text/css, etc.
    if expected_top_level == "text" and actual_top_level == "text":
        return True

    # Historical JavaScript MIME types are still common in captured targets.
    if normalized_expected == "application/javascript":
        if normalized_actual in {
            "text/javascript",
            "application/x-javascript",
            "text/ecmascript",
            "application/ecmascript",
        }:
            return True

    # Rare, but some services emit CSS with an application top-level.
    if normalized_expected == "text/css" and normalized_actual == "application/css":
        return True

    return False


def _expected_content_type_for_kind_and_path(kind: str, path: str) -> str | None:
    lowered = path.lower()
    extension_expected = None
    if lowered == "/robots.txt" or lowered.endswith(".txt"):
        extension_expected = "text/plain"
    elif lowered == "/sitemap.xml" or lowered.endswith(".xml"):
        extension_expected = "application/xml"
    elif lowered.endswith(".json"):
        extension_expected = "application/json"
    elif lowered.endswith(".css"):
        extension_expected = "text/css"
    elif lowered.endswith(".js"):
        extension_expected = "application/javascript"
    elif lowered.endswith((".html", ".htm")):
        extension_expected = "text/html"
    elif lowered.endswith(_BINARY_ASSET_EXTENSIONS):
        extension_expected = _binary_content_type_for_path(path)

    kind_expected_map = {
        "html_page": "text/html",
        "config_text": "text/plain",
        "json_document": "application/json",
        "plain_text": "text/plain",
        "stylesheet": "text/css",
        "javascript": "application/javascript",
        "robots_txt": "text/plain",
        "sitemap_xml": "application/xml",
        "xml_document": "application/xml",
        "credential_bait": "text/plain",
        "log_excerpt": "text/plain",
        "backup_manifest": "text/plain",
        "binary_asset": _binary_content_type_for_path(path),
    }
    kind_expected = kind_expected_map.get(kind)
    kind_overrides_extension = {
        "credential_bait",
        "config_text",
        "log_excerpt",
        "backup_manifest",
        "plain_text",
    }
    if kind in kind_overrides_extension:
        return kind_expected
    return extension_expected or kind_expected


def _test_expected_content_type_for_kind_and_path_cases():
    """Inline smoke-check called from tests; not part of the public API."""
    # server-side script extensions must NOT force text/html — they are dynamic
    assert _expected_content_type_for_kind_and_path("html_page", "/config/database.php") == "text/html", \
        "html_page kind should still expect text/html even without extension hint"
    assert _expected_content_type_for_kind_and_path("config_text", "/config/database.php") == "text/plain", \
        "config_text at .php must return text/plain via kind_overrides_extension"
    assert _expected_content_type_for_kind_and_path("plain_text", "/login.asp") == "text/plain", \
        "plain_text at .asp must return text/plain via kind_overrides_extension"
    # .html/.htm still map to text/html unconditionally
    assert _expected_content_type_for_kind_and_path("html_page", "/index.html") == "text/html"
    assert _expected_content_type_for_kind_and_path("config_text", "/index.html") == "text/plain", \
        "config_text overrides even .html extension"
    # unambiguous extensions still enforced
    assert _expected_content_type_for_kind_and_path("json_document", "/api/data.json") == "application/json"
    assert _expected_content_type_for_kind_and_path("stylesheet", "/assets/main.css") == "text/css"


def _extract_content_type_from_dict_headers(headers: list[dict[str, str]]) -> str | None:
    for header in headers:
        if not isinstance(header, dict):
            continue
        for key, value in header.items():
            if isinstance(key, str) and key.lower() == "content-type" and isinstance(value, str):
                return value
    return None


def _extract_content_type_from_header_hints(header_hints) -> str | None:
    for header_hint in header_hints:
        name = getattr(header_hint, "name", None)
        value = getattr(header_hint, "value", None)
        if isinstance(name, str) and name.lower() == "content-type" and isinstance(value, str):
            return value
    return None


def _planned_output_count(plan: ResourcePlan) -> int:
    return len(plan.artifacts) + len(plan.reference_asset_plan.asset_fetches)


def _collect_planned_paths(plan: ResourcePlan, request: GenerationRequest) -> list[str]:
    artifact_paths = [normalize_path(artifact.path, index_page=request.index_page) for artifact in plan.artifacts]
    asset_paths = [normalize_path(asset.local_path, index_page=request.index_page) for asset in plan.reference_asset_plan.asset_fetches]
    return artifact_paths + asset_paths


def _ensure_unique_paths(paths: Iterable[str]) -> None:
    seen = set()
    for path in paths:
        if path in seen:
            raise ValidationError("duplicate artifact path {}".format(path))
        seen.add(path)


def _ensure_unique_values(values: Iterable[str], label: str) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise ValidationError("duplicate {} {}".format(label, value))
        seen.add(value)


def _normalize_allowed_paths(paths: Iterable[str], request: GenerationRequest) -> set[str]:
    normalized = set()
    for path in paths:
        if isinstance(path, str) and path.strip():
            normalized.add(normalize_path(path, index_page=request.index_page))
    return normalized


def _is_external_reference(reference: str) -> bool:
    parsed = urlsplit(reference)
    if parsed.scheme in {"http", "https"}:
        return True
    if parsed.netloc:
        return True
    return reference.startswith("//")


def _extract_css_urls(value: str) -> list[str]:
    if not isinstance(value, str):
        return []
    refs = []
    for raw in _CSS_URL_RE.findall(value):
        token = raw.strip().strip('"\'')
        if token:
            refs.append(token)
    return refs


def _allowed_baseline_paths(request: GenerationRequest) -> set[str]:
    return {normalize_path(request.index_page, index_page=request.index_page)}


def _validate_local_reference(
    reference: str,
    *,
    field_name: str,
    allowed_paths: set[str],
    request: GenerationRequest,
    forbidden_external_assets: bool,
) -> None:
    if not isinstance(reference, str) or not reference.strip():
        raise ValidationError("{} must be a non-empty path string".format(field_name))

    candidate = reference.strip()
    if _is_external_reference(candidate):
        if forbidden_external_assets:
            raise ValidationError("{} uses external URL {} but external assets are forbidden".format(field_name, candidate))
        return

    normalized = normalize_path(candidate, index_page=request.index_page)
    if normalized not in allowed_paths:
        raise ValidationError("{} references disallowed path {}".format(field_name, normalized))


def validate_plan(plan: ResourcePlan, request: GenerationRequest, runtime_config: GeneratorRuntimeConfig) -> None:
    if not plan.static_only and not runtime_config.enable_scripted_flows:
        raise ValidationError("plan must remain static-only unless scripted flows are enabled")

    planned_output_count = _planned_output_count(plan)
    if planned_output_count > runtime_config.max_bundle_artifacts:
        raise ValidationError("plan exceeds max_bundle_artifacts")

    if plan.bundle_budget_count > runtime_config.max_bundle_artifacts:
        raise ValidationError("plan bundle budget count exceeds runtime limit")

    if plan.bundle_budget_count != planned_output_count:
        raise ValidationError(
            "plan bundle budget count {} must equal planned outputs {}".format(
                plan.bundle_budget_count,
                planned_output_count,
            )
        )

    if plan.bundle_budget_bytes > runtime_config.max_bundle_bytes:
        raise ValidationError("plan bundle budget bytes exceeds runtime limit")

    artifact_ids = [artifact.artifact_id for artifact in plan.artifacts]
    _ensure_unique_values(artifact_ids, "artifact_id")
    asset_ids = [asset_fetch.asset_id for asset_fetch in plan.reference_asset_plan.asset_fetches]
    _ensure_unique_values(asset_ids, "asset_id")

    normalized_paths = _collect_planned_paths(plan, request)
    _ensure_unique_paths(normalized_paths)

    if normalize_path(plan.primary_path, index_page=request.index_page) != request.normalized_path:
        raise ValidationError("plan primary path must match requested normalized path")

    if request.normalized_path not in normalized_paths:
        raise ValidationError("plan must include the primary requested path")

    allowed_primary_kinds = _allowed_kinds_for_path(request.normalized_path)
    if allowed_primary_kinds is not None:
        primary_artifact = next((artifact for artifact in plan.artifacts if artifact.path == request.normalized_path), None)
        if primary_artifact is None:
            raise ValidationError(
                "plan must include a generated primary artifact at {} for extension-enforced kinds {}".format(
                    request.normalized_path,
                    sorted(allowed_primary_kinds),
                )
            )
        if primary_artifact.kind not in allowed_primary_kinds:
            raise ValidationError(
                "primary requested path {} requires artifact kind in {} (got {})".format(
                    request.normalized_path,
                    sorted(allowed_primary_kinds),
                    primary_artifact.kind,
                )
            )
    artifact_id_set = set(artifact_ids)
    for artifact in plan.artifacts:
        validate_planned_artifact(
            artifact,
            request,
            allow_non_static_scopes=runtime_config.enable_scripted_flows,
        )
        unknown_dependencies = [dependency for dependency in artifact.depends_on if dependency not in artifact_id_set]
        if unknown_dependencies:
            raise ValidationError(
                "artifact {} depends on unknown artifacts {}".format(
                    artifact.artifact_id,
                    unknown_dependencies,
                )
            )
        _validate_planned_flow_metadata(artifact, request, set(normalized_paths))
    for asset_fetch in plan.reference_asset_plan.asset_fetches:
        validate_planned_asset_fetch(asset_fetch, request)
        unknown_required = [
            artifact_id for artifact_id in asset_fetch.required_for_artifact_ids if artifact_id not in artifact_id_set
        ]
        if unknown_required:
            raise ValidationError(
                "asset fetch {} references unknown artifact ids {}".format(
                    asset_fetch.asset_id,
                    unknown_required,
                )
            )

    if infer_intent_family(request.normalized_path) == "config_theft":
        if len(plan.artifacts) < 2:
            raise ValidationError("config_theft plans must include at least one supporting artifact")
        if not _has_config_theft_support(plan.artifacts, request.normalized_path):
            raise ValidationError("config_theft plans must include a supporting config/log/backup artifact")

    if infer_intent_family(request.normalized_path) == "cms_probe":
        stylesheet_paths = {artifact.path for artifact in plan.artifacts if artifact.kind == "stylesheet"}
        cms_login_html_artifacts = [
            artifact
            for artifact in plan.artifacts
            if artifact.kind == "html_page" and _is_login_like_path(artifact.path)
        ]
        if cms_login_html_artifacts and not stylesheet_paths:
            raise ValidationError(
                "cms_probe plans with login html artifacts must include at least one stylesheet artifact"
            )
        for artifact in cms_login_html_artifacts:
            linked_paths = {
                normalize_path(link, index_page=request.index_page)
                for link in artifact.links_to
                if isinstance(link, str) and link.strip()
            }
            if not linked_paths.intersection(stylesheet_paths):
                raise ValidationError(
                    "cms_probe login artifact {} must link to a stylesheet artifact via links_to".format(
                        artifact.path
                    )
                )
    if len(plan.artifacts) < 2:
        raise ValidationError("plan must include at least 2 generated artifacts for bundle coherence")


def validate_planned_artifact(
    artifact: PlannedArtifact,
    request: GenerationRequest,
    *,
    allow_non_static_scopes: bool = False,
) -> None:
    allowed_scopes = {"static_file"}
    if allow_non_static_scopes:
        allowed_scopes.update({"dynamic_endpoint", "service_stub"})
    if artifact.artifact_scope not in allowed_scopes:
        raise ValidationError("unsupported artifact scope {}".format(artifact.artifact_scope))
    if artifact.kind == "asset_file":
        raise ValidationError(
            "planned artifact kind asset_file is unsupported for coder generation; use reference_asset_plan.asset_fetches for binary assets"
        )
    if normalize_path(artifact.path, index_page=request.index_page) != artifact.path:
        raise ValidationError("artifact path is not normalized: {}".format(artifact.path))

    expected_content_type = _expected_content_type_for_kind_and_path(artifact.kind, artifact.path)
    contract_content_type = artifact.response_contract.content_type
    if contract_content_type is not None and not _content_type_matches(contract_content_type, expected_content_type):
        raise ValidationError(
            "planned artifact {} has response_contract.content_type {} incompatible with expected {}".format(
                artifact.path,
                contract_content_type,
                expected_content_type,
            )
        )

    hinted_content_type = _extract_content_type_from_header_hints(artifact.response_contract.headers_hint)
    if hinted_content_type is not None and not _content_type_matches(hinted_content_type, contract_content_type or expected_content_type):
        raise ValidationError(
            "planned artifact {} response_contract.headers_hint Content-Type {} conflicts with expected {}".format(
                artifact.path,
                hinted_content_type,
                contract_content_type or expected_content_type,
            )
        )


def _validate_flow_condition_fields(condition, context: str) -> None:
    if condition is None:
        return

    for field_name in ("requires_cookie", "missing_cookie", "requires_prev_path", "method", "requires_header", "missing_header"):
        value = getattr(condition, field_name, None)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValidationError("{} flow condition {} must be a non-empty string".format(context, field_name))

    for field_name in ("query_has", "post_has"):
        values = getattr(condition, field_name, [])
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValidationError("{} flow condition {} contains an empty name".format(context, field_name))

    for field_name in ("header_equals", "header_contains", "query_equals", "query_contains", "post_equals", "post_contains"):
        values = getattr(condition, field_name, {})
        for key, value in values.items():
            if not isinstance(key, str) or not key.strip():
                raise ValidationError("{} flow condition {} contains an empty key".format(context, field_name))
            if not isinstance(value, str):
                raise ValidationError("{} flow condition {} value for {} must be a string".format(context, field_name, key))


def _validate_flow_response_fields(response, context: str, known_paths: set[str] | None = None) -> None:
    if response is None:
        return
    if response.artifact_path is None and response.redirect_to is None:
        raise ValidationError("{} flow response must define artifact_path or redirect_to".format(context))
    if response.artifact_path is not None:
        if not isinstance(response.artifact_path, str) or not response.artifact_path.strip():
            raise ValidationError("{} flow response artifact_path must be non-empty".format(context))
        if known_paths is not None and response.artifact_path not in known_paths:
            raise ValidationError("{} flow response references missing artifact {}".format(context, response.artifact_path))
    if response.redirect_to is not None:
        if not isinstance(response.redirect_to, str) or not response.redirect_to.strip():
            raise ValidationError("{} flow response redirect_to must be non-empty".format(context))
        normalize_path(response.redirect_to)
    for name in response.set_cookie:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("{} flow response set_cookie contains an empty cookie name".format(context))
    for name in response.clear_cookie:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("{} flow response clear_cookie contains an empty cookie name".format(context))
    for header in response.headers:
        if not isinstance(header, dict):
            raise ValidationError("{} flow response headers must be objects".format(context))
        for key, value in header.items():
            if not isinstance(key, str) or not key.strip():
                raise ValidationError("{} flow response header contains an empty name".format(context))
            if not isinstance(value, str):
                raise ValidationError("{} flow response header {} value must be a string".format(context, key))


def _validate_planned_flow_metadata(artifact: PlannedArtifact, request: GenerationRequest, known_paths: set[str]) -> None:
    has_flow_metadata = (
        artifact.flow_match_path is not None
        or artifact.flow_condition is not None
        or artifact.flow_response is not None
    )
    if not has_flow_metadata:
        return
    if not artifact.path.startswith("/_flow/"):
        raise ValidationError("artifact {} defines flow metadata but is not under /_flow/".format(artifact.path))
    if not artifact.flow_match_path:
        raise ValidationError("flow artifact {} requires flow_match_path".format(artifact.path))
    normalized_match_path = normalize_path(artifact.flow_match_path, index_page=request.index_page)
    if normalized_match_path != artifact.flow_match_path:
        raise ValidationError("flow artifact {} flow_match_path is not normalized: {}".format(artifact.path, artifact.flow_match_path))
    _validate_flow_condition_fields(artifact.flow_condition, "planned artifact {}".format(artifact.path))
    if artifact.flow_response is None:
        raise ValidationError("flow artifact {} requires flow_response".format(artifact.path))
    _validate_flow_response_fields(artifact.flow_response, "planned artifact {}".format(artifact.path), known_paths)

def validate_planned_asset_fetch(asset_fetch: PlannedAssetFetch, request: GenerationRequest) -> None:
    if normalize_path(asset_fetch.local_path, index_page=request.index_page) != asset_fetch.local_path:
        raise ValidationError("planned asset local_path is not normalized: {}".format(asset_fetch.local_path))
    if urlsplit(asset_fetch.source_url).scheme not in {"http", "https"}:
        raise ValidationError("planned asset source_url must be http or https")


def validate_artifact_draft(draft: ArtifactDraft, request: GenerationRequest) -> None:
    if normalize_path(draft.path, index_page=request.index_page) != draft.path:
        raise ValidationError("draft path is not normalized: {}".format(draft.path))
    if not isinstance(draft.content_model, dict) or not draft.content_model:
        raise ValidationError("draft content_model must be a non-empty object")
    allowed_kinds = _allowed_kinds_for_path(draft.path)
    if allowed_kinds is not None and draft.kind not in allowed_kinds:
        raise ValidationError(
            "draft path {} requires kind in {} (got {})".format(
                draft.path,
                sorted(allowed_kinds),
                draft.kind,
            )
        )

    if draft.kind == "json_document":
        document = draft.content_model.get("document")
        if not isinstance(document, (dict, list)) or not document:
            raise ValidationError("json_document draft must provide non-empty content_model.document object or array")

    if draft.kind == "plain_text":
        lines = draft.content_model.get("lines")
        if not isinstance(lines, list) or not any(isinstance(line, str) and line.strip() for line in lines):
            raise ValidationError("plain_text draft must provide non-empty content_model.lines")

    if draft.kind == "binary_asset":
        content_type = draft.content_model.get("content_type")
        if not isinstance(content_type, str) or not content_type.strip():
            raise ValidationError("binary_asset draft must provide non-empty content_model.content_type")

        content_base64 = draft.content_model.get("content_base64")
        if not isinstance(content_base64, str) or not content_base64.strip():
            raise ValidationError("binary_asset draft must provide non-empty content_model.content_base64")

        try:
            decoded = base64.b64decode(content_base64, validate=True)
        except Exception as error:
            raise ValidationError("binary_asset draft has invalid base64 payload") from error
        if not decoded:
            raise ValidationError("binary_asset draft base64 payload decodes to empty bytes")


    expected_content_type = _expected_content_type_for_kind_and_path(draft.kind, draft.path)
    if draft.content_type is not None and not _content_type_matches(draft.content_type, expected_content_type):
        raise ValidationError(
            "draft {} content_type {} incompatible with expected {}".format(
                draft.path,
                draft.content_type,
                expected_content_type,
            )
        )

    hinted_content_type = _extract_content_type_from_dict_headers(draft.headers_hint)
    if hinted_content_type is not None and not _content_type_matches(hinted_content_type, draft.content_type or expected_content_type):
        raise ValidationError(
            "draft {} headers_hint Content-Type {} conflicts with expected {}".format(
                draft.path,
                hinted_content_type,
                draft.content_type or expected_content_type,
            )
        )
def validate_artifact_draft_contract(
    draft: ArtifactDraft,
    request: GenerationRequest,
    *,
    allowed_local_asset_paths: list[str],
    allowed_internal_paths: list[str],
    primary_path: str,
    forbidden_external_assets: bool,
) -> None:
    allowed_local = _normalize_allowed_paths(allowed_local_asset_paths, request)
    allowed_internal = _normalize_allowed_paths(allowed_internal_paths, request)
    allowed_internal.add(normalize_path(primary_path, index_page=request.index_page))
    allowed_paths = allowed_local | allowed_internal

    if not allowed_paths:
        raise ValidationError("artifact contract has no allowed paths")

    if draft.kind == "html_page":
        model = draft.content_model
        for index, stylesheet in enumerate(model.get("linked_stylesheets", [])):
            _validate_local_reference(
                stylesheet,
                field_name="linked_stylesheets[{}]".format(index),
                allowed_paths=allowed_paths,
                request=request,
                forbidden_external_assets=forbidden_external_assets,
            )
        for index, script in enumerate(model.get("linked_scripts", [])):
            _validate_local_reference(
                script,
                field_name="linked_scripts[{}]".format(index),
                allowed_paths=allowed_paths,
                request=request,
                forbidden_external_assets=forbidden_external_assets,
            )
        for index, image in enumerate(model.get("images", [])):
            if not isinstance(image, dict):
                continue
            _validate_local_reference(
                image.get("src", ""),
                field_name="images[{}].src".format(index),
                allowed_paths=allowed_paths,
                request=request,
                forbidden_external_assets=forbidden_external_assets,
            )
            href = image.get("href")
            if isinstance(href, str) and href.strip():
                _validate_local_reference(
                    href,
                    field_name="images[{}].href".format(index),
                    allowed_paths=allowed_internal,
                    request=request,
                    forbidden_external_assets=forbidden_external_assets,
                )
        for index, link in enumerate(model.get("nav_links", [])):
            if not isinstance(link, dict):
                continue
            _validate_local_reference(
                link.get("href", ""),
                field_name="nav_links[{}].href".format(index),
                allowed_paths=allowed_internal,
                request=request,
                forbidden_external_assets=forbidden_external_assets,
            )
        form = model.get("form")
        if isinstance(form, dict) and isinstance(form.get("action"), str):
            _validate_local_reference(
                form.get("action", ""),
                field_name="form.action",
                allowed_paths=allowed_internal,
                request=request,
                forbidden_external_assets=forbidden_external_assets,
            )

    if draft.kind == "stylesheet":
        for index, rule in enumerate(draft.content_model.get("rules", [])):
            if not isinstance(rule, dict):
                continue
            declarations = rule.get("declarations", {})
            if not isinstance(declarations, dict):
                continue
            for property_name, value in declarations.items():
                if not isinstance(value, str):
                    continue
                for url_reference in _extract_css_urls(value):
                    _validate_local_reference(
                        url_reference,
                        field_name="rules[{}].{}".format(index, property_name),
                        allowed_paths=allowed_paths,
                        request=request,
                        forbidden_external_assets=forbidden_external_assets,
                    )

    if draft.kind == "javascript":
        for index, line in enumerate(draft.content_model.get("lines", [])):
            if not isinstance(line, str):
                continue
            for external_url in _JS_EXTERNAL_URL_RE.findall(line):
                if forbidden_external_assets:
                    raise ValidationError(
                        "javascript line {} uses external URL {} but external assets are forbidden".format(index, external_url)
                    )
            for path_literal in _JS_PATH_LITERAL_RE.findall(line):
                _validate_local_reference(
                    path_literal,
                    field_name="javascript.lines[{}]".format(index),
                    allowed_paths=allowed_paths,
                    request=request,
                    forbidden_external_assets=forbidden_external_assets,
                )


def validate_generated_artifact(
    artifact: GeneratedArtifact,
    request: GenerationRequest,
    *,
    allow_non_static_scopes: bool = False,
) -> None:
    if normalize_path(artifact.path, index_page=request.index_page) != artifact.path:
        raise ValidationError("generated artifact path is not normalized: {}".format(artifact.path))
    allowed_scopes = {"static_file"}
    if allow_non_static_scopes:
        allowed_scopes.update({"dynamic_endpoint", "service_stub"})
    if artifact.artifact_scope not in allowed_scopes:
        raise ValidationError("generated artifact scope {} is unsupported".format(artifact.artifact_scope))
    if artifact.status_code < 100 or artifact.status_code > 599:
        raise ValidationError("generated artifact status code is out of range")
    if not artifact.body_bytes:
        raise ValidationError("generated artifact body is empty")

    content_type_header = _extract_content_type_from_dict_headers(artifact.headers)
    if content_type_header is None:
        raise ValidationError("generated artifact missing content-type header")

    expected_content_type = _expected_content_type_for_kind_and_path(artifact.kind, artifact.path)
    if not _content_type_matches(content_type_header, expected_content_type):
        raise ValidationError(
            "generated artifact {} Content-Type {} is incompatible with expected {}".format(
                artifact.path,
                content_type_header,
                expected_content_type,
            )
        )

    if artifact.kind != "binary_asset":
        decoded_body = artifact.body_bytes.decode("utf-8", errors="ignore")
        internal_term_match = _INTERNAL_LANGUAGE_RE.search(decoded_body)
        if internal_term_match is not None:
            raise ValidationError(
                "generated artifact leaked internal planning language: {}".format(internal_term_match.group(1))
            )


def extract_html_references(body: bytes) -> list[str]:
    html_text = body.decode("utf-8", errors="ignore")
    return [reference for reference in _LINK_RE.findall(html_text) if isinstance(reference, str) and reference.strip()]


def extract_css_references(body: bytes) -> list[str]:
    css_text = body.decode("utf-8", errors="ignore")
    references = []
    for value in _CSS_URL_RE.findall(css_text):
        token = value.strip().strip('"\'')
        if token:
            references.append(token)
    return references


def extract_javascript_references(body: bytes) -> list[str]:
    script_text = body.decode("utf-8", errors="ignore")
    references = []
    references.extend(_JS_EXTERNAL_URL_RE.findall(script_text))
    references.extend(_JS_PATH_LITERAL_RE.findall(script_text))
    return references


def extract_internal_links(body: bytes) -> list[str]:
    links = []
    for reference in extract_html_references(body):
        if _is_external_reference(reference):
            continue
        if reference.startswith(("mailto:", "javascript:", "#", "data:")):
            continue
        links.append(normalize_path(reference))
    return links


def _validate_bundle_reference(
    *,
    reference: str,
    source_artifact_path: str,
    reference_kind: str,
    allowed_paths: set[str],
    request: GenerationRequest,
    forbidden_external_assets: bool,
) -> None:
    candidate = reference.strip()
    if not candidate or candidate.startswith(("#", "mailto:", "javascript:", "data:")):
        return

    if _is_external_reference(candidate):
        if forbidden_external_assets:
            raise ValidationError(
                "artifact {} {} uses forbidden external URL {}".format(source_artifact_path, reference_kind, candidate)
            )
        return

    normalized = normalize_path(candidate, index_page=request.index_page)
    if normalized not in allowed_paths:
        raise ValidationError(
            "artifact {} {} references missing path {}".format(source_artifact_path, reference_kind, normalized)
        )




def validate_flow_descriptor(descriptor, bundle_paths: set) -> None:
    """
    Check a FlowDescriptor for consistency against the artifact paths in a bundle.

    Raises ValidationError if:
    - Any rule references an artifact_path not present in the bundle
    - A rule has neither artifact_path nor redirect_to
    - A redirect_to is not a valid normalised path
    - A cookie name is empty
    """
    for rule in descriptor.rules:
        if not isinstance(rule.match_path, str) or not rule.match_path.strip():
            raise ValidationError("flow rule has empty match_path")
        normalized_match_path = normalize_path(rule.match_path)
        if normalized_match_path != rule.match_path:
            raise ValidationError("flow rule match_path is not normalized: {}".format(rule.match_path))
        _validate_flow_condition_fields(rule.condition, "flow rule for {!r}".format(rule.match_path))
        _validate_flow_response_fields(
            rule.response,
            "flow rule for {!r}".format(rule.match_path),
            bundle_paths,
        )


def _flow_condition_signature(condition) -> str:
    if condition is None:
        return "{}"
    return json.dumps(
        condition.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
        sort_keys=True,
        separators=(",", ":"),
    )


def _flow_response_label(rule) -> str:
    if rule.response.artifact_path:
        return "artifact_path={}".format(rule.response.artifact_path)
    if rule.response.redirect_to:
        return "redirect_to={}".format(rule.response.redirect_to)
    return "empty_response"


def diagnose_flow_reachability(bundle: GeneratedBundle) -> list[str]:
    """
    Return non-blocking diagnostics for V2 flow artifacts/rules that are likely
    unreachable. These diagnostics are intentionally warnings: generated content
    should still be persisted so the failed flow shape can be inspected later.
    """
    descriptor = getattr(bundle, "flow_descriptor", None)
    if descriptor is None:
        return []

    bundle_paths = {artifact.path for artifact in bundle.artifacts}
    flow_artifact_paths = {path for path in bundle_paths if path.startswith("/_flow/")}
    referenced_artifact_paths = {
        rule.response.artifact_path
        for rule in descriptor.rules
        if rule.response.artifact_path is not None
    }
    sourcing_artifact_paths = {
        rule.source_artifact_path
        for rule in descriptor.rules
        if rule.source_artifact_path is not None
    }
    tied_in_artifact_paths = referenced_artifact_paths | sourcing_artifact_paths

    diagnostics: list[str] = []
    for path in sorted(flow_artifact_paths - tied_in_artifact_paths):
        diagnostics.append(
            "flow artifact {} is not served by any flow rule".format(path)
        )

    sorted_rules = sorted(
        enumerate(descriptor.rules),
        key=lambda item: (-item[1].priority, item[0]),
    )
    first_rule_by_condition: dict[tuple[str, str], tuple[int, int, str]] = {}
    for original_index, rule in sorted_rules:
        key = (rule.match_path, _flow_condition_signature(rule.condition))
        current_label = _flow_response_label(rule)
        if key in first_rule_by_condition:
            prior_index, prior_priority, prior_label = first_rule_by_condition[key]
            diagnostics.append(
                "flow rule #{} for {} is shadowed by earlier rule #{} with the same match_path/condition "
                "(priority {} {}, priority {} {})".format(
                    original_index + 1,
                    rule.match_path,
                    prior_index + 1,
                    prior_priority,
                    prior_label,
                    rule.priority,
                    current_label,
                )
            )
            continue
        first_rule_by_condition[key] = (original_index, rule.priority, current_label)

    return diagnostics
def validate_bundle(bundle: GeneratedBundle, request: GenerationRequest, runtime_config: GeneratorRuntimeConfig) -> None:
    normalized_primary = normalize_path(bundle.primary_path, index_page=request.index_page)
    if normalized_primary != request.normalized_path:
        raise ValidationError("bundle primary path must match requested normalized path")


    _ensure_unique_paths(artifact.path for artifact in bundle.artifacts)

    total_bytes = 0
    available_paths = set()
    artifact_by_path = {artifact.path: artifact for artifact in bundle.artifacts}
    for artifact in bundle.artifacts:
        validate_generated_artifact(
            artifact,
            request,
            allow_non_static_scopes=runtime_config.enable_scripted_flows,
        )
        total_bytes += len(artifact.body_bytes)
        available_paths.add(artifact.path)

    if total_bytes > runtime_config.max_bundle_bytes:
        raise ValidationError("bundle exceeds max_bundle_bytes")

    if request.normalized_path not in available_paths:
        raise ValidationError("bundle is missing the primary requested artifact")
    flow_descriptor = getattr(bundle, "flow_descriptor", None)
    allowed_primary_kinds = _allowed_kinds_for_path(request.normalized_path)
    if allowed_primary_kinds is not None:
        primary_artifact = next((artifact for artifact in bundle.artifacts if artifact.path == request.normalized_path), None)
        if primary_artifact is None:
            raise ValidationError("bundle is missing the primary requested artifact")
        if primary_artifact.kind not in allowed_primary_kinds:
            raise ValidationError(
                "bundle primary artifact {} requires kind in {} (got {})".format(
                    request.normalized_path,
                    sorted(allowed_primary_kinds),
                    primary_artifact.kind,
                )
            )
    allowed_paths = available_paths | _allowed_baseline_paths(request)

    login_post_targets = _collect_login_post_targets(bundle, request)
    if login_post_targets and runtime_config.enable_scripted_flows:
        if flow_descriptor is None:
            raise ValidationError(
                "POST forms require a flow descriptor with a three-attempt failure loop and one-minute lockout"
            )
        validate_flow_descriptor(flow_descriptor, available_paths)
        _validate_login_flow_rules(flow_descriptor, login_post_targets, artifact_by_path)
    elif flow_descriptor is not None:
        validate_flow_descriptor(flow_descriptor, available_paths)

    if flow_descriptor is not None:
        for rule in flow_descriptor.rules:
            redirect_to = rule.response.redirect_to
            if redirect_to is None:
                continue
            normalized_redirect = normalize_path(redirect_to, index_page=request.index_page)
            if normalized_redirect not in allowed_paths:
                raise ValidationError(
                    "flow rule redirect_to {!r} references missing path {}".format(
                        redirect_to,
                        normalized_redirect,
                    )
                )

    for artifact in bundle.artifacts:
        if artifact.kind == "html_page":
            for reference in extract_html_references(artifact.body_bytes):
                _validate_bundle_reference(
                    reference=reference,
                    source_artifact_path=artifact.path,
                    reference_kind="html reference",
                    allowed_paths=allowed_paths,
                    request=request,
                    forbidden_external_assets=True,
                )
        elif artifact.kind == "stylesheet":
            for reference in extract_css_references(artifact.body_bytes):
                _validate_bundle_reference(
                    reference=reference,
                    source_artifact_path=artifact.path,
                    reference_kind="css url()",
                    allowed_paths=allowed_paths,
                    request=request,
                    forbidden_external_assets=True,
                )
        elif artifact.kind == "javascript":
            for reference in extract_javascript_references(artifact.body_bytes):
                _validate_bundle_reference(
                    reference=reference,
                    source_artifact_path=artifact.path,
                    reference_kind="javascript path",
                    allowed_paths=allowed_paths,
                    request=request,
                    forbidden_external_assets=True,
                )

    if infer_intent_family(request.normalized_path) == "config_theft":
        if len(bundle.artifacts) < 2:
            raise ValidationError("config_theft bundles must include at least one supporting artifact")
        if not _has_config_theft_support(bundle.artifacts, request.normalized_path):
            raise ValidationError("config_theft bundles must include a supporting config/log/backup artifact")
