from __future__ import annotations

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


def _is_cms_login_html_path(path: str) -> bool:
    lowered = path.lower()
    return lowered == "/wp-login.php" or (lowered.startswith("/wp-admin/") and "login" in lowered)


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
    if not plan.static_only:
        raise ValidationError("plan must remain static-only in v1")

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

    artifact_id_set = set(artifact_ids)
    for artifact in plan.artifacts:
        validate_planned_artifact(artifact, request)
        unknown_dependencies = [dependency for dependency in artifact.depends_on if dependency not in artifact_id_set]
        if unknown_dependencies:
            raise ValidationError(
                "artifact {} depends on unknown artifacts {}".format(
                    artifact.artifact_id,
                    unknown_dependencies,
                )
            )
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
            if artifact.kind == "html_page" and _is_cms_login_html_path(artifact.path)
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


def validate_planned_artifact(artifact: PlannedArtifact, request: GenerationRequest) -> None:
    if artifact.artifact_scope != "static_file":
        raise ValidationError("unsupported artifact scope {}".format(artifact.artifact_scope))
    if normalize_path(artifact.path, index_page=request.index_page) != artifact.path:
        raise ValidationError("artifact path is not normalized: {}".format(artifact.path))


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


def validate_generated_artifact(artifact: GeneratedArtifact, request: GenerationRequest) -> None:
    if normalize_path(artifact.path, index_page=request.index_page) != artifact.path:
        raise ValidationError("generated artifact path is not normalized: {}".format(artifact.path))
    if artifact.artifact_scope != "static_file":
        raise ValidationError("generated artifact is not static")
    if artifact.status_code != 200:
        raise ValidationError("generated artifact status code must be 200 for static v1")
    if not artifact.body_bytes:
        raise ValidationError("generated artifact body is empty")
    if not any(
        isinstance(header, dict) and any(key.lower() == "content-type" for key in header.keys()) for header in artifact.headers
    ):
        raise ValidationError("generated artifact missing content-type header")

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


def validate_bundle(bundle: GeneratedBundle, request: GenerationRequest, runtime_config: GeneratorRuntimeConfig) -> None:
    normalized_primary = normalize_path(bundle.primary_path, index_page=request.index_page)
    if normalized_primary != request.normalized_path:
        raise ValidationError("bundle primary path must match requested normalized path")

    if len(bundle.artifacts) > runtime_config.max_bundle_artifacts:
        raise ValidationError("bundle exceeds max_bundle_artifacts")

    _ensure_unique_paths(artifact.path for artifact in bundle.artifacts)

    total_bytes = 0
    available_paths = set()
    for artifact in bundle.artifacts:
        validate_generated_artifact(artifact, request)
        total_bytes += len(artifact.body_bytes)
        available_paths.add(artifact.path)

    if total_bytes > runtime_config.max_bundle_bytes:
        raise ValidationError("bundle exceeds max_bundle_bytes")

    if request.normalized_path not in available_paths:
        raise ValidationError("bundle is missing the primary requested artifact")

    allowed_paths = available_paths | _allowed_baseline_paths(request)
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
