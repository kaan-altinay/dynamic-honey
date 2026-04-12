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
    ResourcePlan,
)


_LINK_RE = re.compile(r'(?:href|src)=["\']([^"\']+)["\']', re.I)
_CONFIG_THEFT_SUPPORT_KINDS = {"config_text", "log_excerpt", "backup_manifest", "credential_bait"}
_INTERNAL_LANGUAGE_RE = re.compile(r"\b(fake|lure|attacker|attackers|honeypot)\b", re.I)


def _has_config_theft_support(artifacts, primary_path: str) -> bool:
    for artifact in artifacts:
        if getattr(artifact, "path", None) == primary_path:
            continue
        if getattr(artifact, "kind", None) in _CONFIG_THEFT_SUPPORT_KINDS:
            return True
    return False


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


def _ensure_unique_paths(paths: Iterable[str]) -> None:
    seen = set()
    for path in paths:
        if path in seen:
            raise ValidationError("duplicate artifact path {}".format(path))
        seen.add(path)


def validate_plan(plan: ResourcePlan, request: GenerationRequest, runtime_config: GeneratorRuntimeConfig) -> None:
    if not plan.static_only:
        raise ValidationError("plan must remain static-only in v1")

    if len(plan.artifacts) > runtime_config.max_bundle_artifacts:
        raise ValidationError("plan exceeds max_bundle_artifacts")

    if plan.bundle_budget_count > runtime_config.max_bundle_artifacts:
        raise ValidationError("plan bundle budget count exceeds runtime limit")

    normalized_paths = [normalize_path(artifact.path, index_page=request.index_page) for artifact in plan.artifacts]
    _ensure_unique_paths(normalized_paths)

    if normalize_path(plan.primary_path, index_page=request.index_page) != request.normalized_path:
        raise ValidationError("plan primary path must match requested normalized path")

    if request.normalized_path not in normalized_paths:
        raise ValidationError("plan must include the primary requested path")

    for artifact in plan.artifacts:
        validate_planned_artifact(artifact, request)


    if infer_intent_family(request.normalized_path) == "config_theft":
        if len(plan.artifacts) < 2:
            raise ValidationError("config_theft plans must include at least one supporting artifact")
        if not _has_config_theft_support(plan.artifacts, request.normalized_path):
            raise ValidationError("config_theft plans must include a supporting config/log/backup artifact")

def validate_planned_artifact(artifact: PlannedArtifact, request: GenerationRequest) -> None:
    if artifact.artifact_scope != "static_file":
        raise ValidationError("unsupported artifact scope {}".format(artifact.artifact_scope))
    if normalize_path(artifact.path, index_page=request.index_page) != artifact.path:
        raise ValidationError("artifact path is not normalized: {}".format(artifact.path))


def validate_artifact_draft(draft: ArtifactDraft, request: GenerationRequest) -> None:
    if normalize_path(draft.path, index_page=request.index_page) != draft.path:
        raise ValidationError("draft path is not normalized: {}".format(draft.path))
    if not isinstance(draft.content_model, dict) or not draft.content_model:
        raise ValidationError("draft content_model must be a non-empty object")


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


def extract_internal_links(body: bytes) -> list[str]:
    html_text = body.decode("utf-8", errors="ignore")
    links = []
    for link in _LINK_RE.findall(html_text):
        if not link or link.startswith(("http://", "https://", "mailto:", "javascript:", "#")):
            continue
        links.append(normalize_path(link))
    return links


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

    for artifact in bundle.artifacts:
        for linked_path in extract_internal_links(artifact.body_bytes):
            if linked_path not in available_paths:
                raise ValidationError("bundle contains broken internal link {}".format(linked_path))

    if infer_intent_family(request.normalized_path) == "config_theft":
        if len(bundle.artifacts) < 2:
            raise ValidationError("config_theft bundles must include at least one supporting artifact")
        if not _has_config_theft_support(bundle.artifacts, request.normalized_path):
            raise ValidationError("config_theft bundles must include a supporting config/log/backup artifact")
