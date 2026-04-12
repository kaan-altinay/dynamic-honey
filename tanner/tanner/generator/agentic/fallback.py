from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import Any

from tanner.generator.agentic.models import ArtifactDraft, ExpertSpec, GeneratedBundle, GenerationRequest
from tanner.generator.agentic.renderers import render_artifact
from tanner.generator.agentic.validators import infer_intent_family


@dataclass(frozen=True)
class FallbackArtifactBlueprint:
    artifact_id: str
    path: str
    kind: str
    content_model: dict[str, Any]
    review_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FallbackProfile:
    variables: dict[str, str] = field(default_factory=dict)
    artifacts: tuple[FallbackArtifactBlueprint, ...] = ()


def _backup_variant(primary_path: str) -> str:
    return primary_path + ".bak" if not primary_path.endswith(".bak") else primary_path + ".old"


def _substitute_templates(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return Template(value).safe_substitute(variables)
    if isinstance(value, list):
        return [_substitute_templates(item, variables) for item in value]
    if isinstance(value, tuple):
        return tuple(_substitute_templates(item, variables) for item in value)
    if isinstance(value, dict):
        return {key: _substitute_templates(item, variables) for key, item in value.items()}
    return value


def _resolve_profile_variables(request: GenerationRequest, profile: FallbackProfile) -> dict[str, str]:
    variables = {
        "primary_path": request.normalized_path,
        "index_page": request.index_page,
        "backup_path": _backup_variant(request.normalized_path),
    }
    for key, value in profile.variables.items():
        variables[key] = Template(value).safe_substitute(variables)
    return variables


def _build_profile_drafts(request: GenerationRequest, profile_name: str) -> list[ArtifactDraft]:
    profile = FALLBACK_PROFILES[profile_name]
    variables = _resolve_profile_variables(request, profile)
    drafts = []
    for blueprint in profile.artifacts:
        drafts.append(
            ArtifactDraft(
                artifact_id=blueprint.artifact_id,
                path=_substitute_templates(blueprint.path, variables),
                kind=blueprint.kind,
                content_model=_substitute_templates(blueprint.content_model, variables),
                review_notes=list(blueprint.review_notes) or ["fallback profile {}".format(profile_name)],
            )
        )
    return drafts

def _fit_drafts_to_budget(drafts: list[ArtifactDraft], max_artifacts: int | None) -> list[ArtifactDraft]:
    if max_artifacts is None or len(drafts) <= max_artifacts:
        return drafts

    retained_drafts = [draft.model_copy(deep=True) for draft in drafts[:max_artifacts]]
    retained_paths = {draft.path for draft in retained_drafts}
    budgeted_drafts = []

    for draft in retained_drafts:
        content_model = dict(draft.content_model)
        if draft.kind == "html_page":
            content_model["linked_stylesheets"] = [
                path for path in content_model.get("linked_stylesheets", []) if path in retained_paths
            ]
            content_model["linked_scripts"] = [
                path for path in content_model.get("linked_scripts", []) if path in retained_paths
            ]
            content_model["nav_links"] = [
                link
                for link in content_model.get("nav_links", [])
                if isinstance(link, dict) and link.get("href") in retained_paths
            ]
            form = content_model.get("form")
            if isinstance(form, dict):
                action = form.get("action")
                if isinstance(action, str) and action.startswith("/") and action not in retained_paths:
                    adjusted_form = dict(form)
                    adjusted_form["action"] = draft.path
                    content_model["form"] = adjusted_form
        elif draft.kind == "sitemap_xml":
            content_model["urls"] = [url for url in content_model.get("urls", []) if url in retained_paths]

        budgeted_drafts.append(draft.model_copy(update={"content_model": content_model}))

    return budgeted_drafts


FALLBACK_PROFILES: dict[str, FallbackProfile] = {
    "cms_probe": FallbackProfile(
        variables={
            "stylesheet_path": "/wp-content/themes/twentytwenty/style.css",
            "script_path": "/wp-includes/js/wp-login.js",
            "helper_path": "/wp-login.php",
        },
        artifacts=(
            FallbackArtifactBlueprint(
                artifact_id="wp-login",
                path="${primary_path}",
                kind="html_page",
                content_model={
                    "title": "WordPress Login",
                    "heading": "Log In",
                    "paragraphs": [
                        "Use your WordPress account credentials to continue.",
                        "Session activity is monitored for administrative review.",
                    ],
                    "linked_stylesheets": ["${stylesheet_path}"],
                    "linked_scripts": ["${script_path}"],
                    "nav_links": [{"label": "Lost Password", "href": "${helper_path}"}],
                    "form": {
                        "action": "${primary_path}",
                        "method": "post",
                        "fields": [
                            {"name": "log", "label": "Username or Email", "type": "text"},
                            {"name": "pwd", "label": "Password", "type": "password"},
                        ],
                        "submit_label": "Log In",
                    },
                    "footer": "WordPress administrative access portal",
                },
            ),
            FallbackArtifactBlueprint(
                artifact_id="wp-style",
                path="${stylesheet_path}",
                kind="stylesheet",
                content_model={
                    "rules": [
                        {"selector": "body", "declarations": {"font-family": "Arial, sans-serif", "background": "#f0f0f1", "color": "#1d2327"}},
                        {"selector": "main", "declarations": {"max-width": "420px", "margin": "4rem auto", "padding": "2rem", "background": "#fff", "border": "1px solid #c3c4c7"}},
                        {"selector": "label", "declarations": {"display": "block", "margin-bottom": "1rem"}},
                    ]
                },
            ),
            FallbackArtifactBlueprint(
                artifact_id="wp-script",
                path="${script_path}",
                kind="javascript",
                content_model={
                    "lines": [
                        "document.addEventListener('DOMContentLoaded', function () {",
                        "  var firstInput = document.querySelector('input[name=\"log\"]');",
                        "  if (firstInput) { firstInput.focus(); }",
                        "});",
                    ]
                },
            ),
            FallbackArtifactBlueprint(
                artifact_id="wp-lost-password",
                path="${helper_path}",
                kind="html_page",
                content_model={
                    "title": "Password Reset",
                    "heading": "Reset Your Password",
                    "paragraphs": ["Enter your username or email address to receive a password reset link."],
                    "linked_stylesheets": ["${stylesheet_path}"],
                    "nav_links": [{"label": "Back to login", "href": "${primary_path}"}],
                    "form": {
                        "action": "${helper_path}",
                        "method": "post",
                        "fields": [{"name": "user_login", "label": "Username or Email", "type": "text"}],
                        "submit_label": "Get New Password",
                    },
                    "footer": "WordPress password recovery",
                },
            ),
        ),
    ),
    "config_theft": FallbackProfile(
        variables={
            "log_path": "/storage/logs/app.log",
        },
        artifacts=(
            FallbackArtifactBlueprint(
                artifact_id="primary-config",
                path="${primary_path}",
                kind="config_text",
                content_model={
                    "format": "env",
                    "comment": "Application environment configuration",
                    "entries": [
                        {"key": "APP_NAME", "value": "production-portal"},
                        {"key": "APP_ENV", "value": "production"},
                        {"key": "APP_KEY", "value": "base64:O0vJm0QW2N7PkQemv3a8sB2sS4oI1C4e"},
                        {"key": "APP_URL", "value": "https://portal.example.internal"},
                        {"key": "APP_DEBUG", "value": "false"},
                        {"key": "CACHE_DRIVER", "value": "redis"},
                        {"key": "SESSION_DRIVER", "value": "redis"},
                        {"key": "QUEUE_CONNECTION", "value": "database"},
                        {"key": "REDIS_HOST", "value": "10.24.18.16"},
                        {"key": "DB_HOST", "value": "10.24.18.12"},
                        {"key": "DB_DATABASE", "value": "billing"},
                        {"key": "DB_USERNAME", "value": "svc_portal"},
                        {"key": "DB_PASSWORD", "value": "P@ssw0rd!2026"},
                        {"key": "MAIL_HOST", "value": "smtp.internal.example"},
                        {"key": "MAIL_PORT", "value": "587"},
                        {"key": "MAIL_USERNAME", "value": "mailer@internal.example"},
                        {"key": "MAIL_PASSWORD", "value": "M4ilP@ss!2026"},
                    ],
                },
            ),
            FallbackArtifactBlueprint(
                artifact_id="config-log",
                path="${log_path}",
                kind="log_excerpt",
                content_model={
                    "lines": [
                        "[2026-04-03 00:13:12] production.INFO: refresh config cache completed",
                        "[2026-04-03 00:13:17] production.WARNING: database auth retry for svc_portal against 10.24.18.12",
                        "[2026-04-03 00:13:21] production.INFO: secrets loaded from ${primary_path}",
                    ]
                },
            ),
            FallbackArtifactBlueprint(
                artifact_id="config-backup",
                path="${backup_path}",
                kind="backup_manifest",
                content_model={
                    "lines": [
                        "manifest-version: 1",
                        "snapshot: nightly-2026-04-03",
                        "include: ${primary_path}",
                        "include: ${log_path}",
                    ]
                },
            ),
        ),
    ),
    "generic_recon": FallbackProfile(
        artifacts=(
            FallbackArtifactBlueprint(
                artifact_id="primary-page",
                path="${primary_path}",
                kind="html_page",
                content_model={
                    "title": "Service Portal",
                    "heading": "Service Portal",
                    "paragraphs": [
                        "The requested resource is available from this service bundle.",
                        "Supporting files are published to preserve a realistic navigation structure.",
                    ],
                    "nav_links": [{"label": "Robots", "href": "/robots.txt"}],
                    "footer": "Service portal resources",
                },
            ),
            FallbackArtifactBlueprint(
                artifact_id="robots",
                path="/robots.txt",
                kind="robots_txt",
                content_model={"lines": ["User-agent: *", "Disallow: /private", "Disallow: /admin"]},
            ),
        ),
    ),
}

INTENT_TO_FALLBACK_PROFILE = {
    "cms_probe": "cms_probe",
    "config_theft": "config_theft",
}


def build_fallback_bundle(
    request: GenerationRequest,
    expert_spec: ExpertSpec | None = None,
    reasons: list[str] | None = None,
    max_artifacts: int | None = None,
 ) -> GeneratedBundle:
    intent_family = expert_spec.intent_family if expert_spec is not None else infer_intent_family(request.normalized_path)
    profile_name = INTENT_TO_FALLBACK_PROFILE.get(intent_family, "generic_recon")
    drafts = _build_profile_drafts(request, profile_name)
    drafts = _fit_drafts_to_budget(drafts, max_artifacts)
    artifacts = [render_artifact(draft) for draft in drafts]
    review_bits = reasons or ["used deterministic fallback profile {}".format(profile_name)]
    return GeneratedBundle(
        primary_path=request.normalized_path,
        artifacts=artifacts,
        review_summary="; ".join(review_bits),
        used_fallback=True,
    )
