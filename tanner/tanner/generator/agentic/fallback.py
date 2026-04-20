from __future__ import annotations

import base64
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


def _required_kind_for_path(path: str) -> str | None:
    lowered = path.lower()
    if lowered == "/robots.txt":
        return "robots_txt"
    if lowered == "/sitemap.xml":
        return "sitemap_xml"
    if lowered.endswith(".xml"):
        return "xml_document"
    if lowered.endswith(".json"):
        return "json_document"
    if lowered.endswith(".txt"):
        return "plain_text"
    if lowered.endswith((".ico", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".woff", ".woff2", ".ttf", ".otf")):
        return "binary_asset"
    return None


def _binary_asset_content_type_for_path(path: str) -> str:
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
    return "application/octet-stream"


def _binary_asset_stub_base64(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".png") or lowered.endswith(".ico"):
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7bRz4AAAAASUVORK5CYII="
    if lowered.endswith((".jpg", ".jpeg")):
        return "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBUQEBAVFRUVFRUVFRUVFRUVFRUVFRUXFhUVFRUYHSggGBolGxUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OFQ8QFS0dHR0tLS0rKy0tLSstKy0rKy0tLS0tLS0tLS0tLS0tLSstLS0tLS0tLS0tKy0tLS0tK//AABEIAAEAAQMBIgACEQEDEQH/xAAXAAEAAwAAAAAAAAAAAAAAAAAAAAUG/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEAMQAAAB9A//xAAZEAEAAwEBAAAAAAAAAAAAAAABAAIRITH/2gAIAQEAAT8A0YxW4VxYf//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8Af//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8Af//Z"
    if lowered.endswith(".gif"):
        return "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
    if lowered.endswith(".svg"):
        svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\"></svg>"
        return base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return base64.b64encode(b"binary-asset").decode("ascii")


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


def _build_xml_fallback_drafts(request: GenerationRequest) -> list[ArtifactDraft]:
    primary_path = request.normalized_path
    support_candidates = ["/WANCfgSCPD.xml", "/WANIPConnSCPD.xml"]
    support_paths = [candidate for candidate in support_candidates if candidate != primary_path]
    if not support_paths:
        support_paths = ["/device.xml"]

    primary_lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<root xmlns=\"urn:schemas-upnp-org:device-1-0\">",
        "  <specVersion><major>1</major><minor>0</minor></specVersion>",
        "  <URLBase>http://10.44.12.1:1900/</URLBase>",
        "  <device>",
        "    <friendlyName>Northbridge Branch Gateway</friendlyName>",
        "    <modelName>{}</modelName>".format(primary_path.rsplit("/", 1)[-1] or "descriptor.xml"),
        "    <serviceList>",
        "      <service>",
        "        <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>",
        "        <SCPDURL>{}</SCPDURL>".format(support_paths[0]),
        "      </service>",
        "    </serviceList>",
        "  </device>",
        "</root>",
    ]

    drafts = [
        ArtifactDraft(
            artifact_id="xml-primary",
            path=primary_path,
            kind="xml_document",
            content_model={"lines": primary_lines},
            review_notes=["fallback profile xml_recon"],
        ),
        ArtifactDraft(
            artifact_id="xml-support-a",
            path=support_paths[0],
            kind="xml_document",
            content_model={
                "lines": [
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                    "<scpd xmlns=\"urn:schemas-upnp-org:service-1-0\">",
                    "  <specVersion><major>1</major><minor>0</minor></specVersion>",
                    "  <actionList><action><name>GetStatusInfo</name></action></actionList>",
                    "</scpd>",
                ]
            },
            review_notes=["fallback profile xml_recon"],
        ),
    ]

    if len(support_paths) > 1:
        drafts.append(
            ArtifactDraft(
                artifact_id="xml-support-b",
                path=support_paths[1],
                kind="xml_document",
                content_model={
                    "lines": [
                        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                        "<scpd xmlns=\"urn:schemas-upnp-org:service-1-0\">",
                        "  <specVersion><major>1</major><minor>0</minor></specVersion>",
                        "  <actionList><action><name>GetExternalIPAddress</name></action></actionList>",
                        "</scpd>",
                    ]
                },
                review_notes=["fallback profile xml_recon"],
            )
        )
    return drafts


def _build_json_fallback_drafts(request: GenerationRequest) -> list[ArtifactDraft]:
    primary_path = request.normalized_path
    support_path = "/version" if primary_path != "/version" else "/info"
    return [
        ArtifactDraft(
            artifact_id="json-primary",
            path=primary_path,
            kind="json_document",
            content_model={
                "document": {
                    "status": "ok",
                    "path": primary_path,
                    "service": "fallback-json",
                }
            },
            review_notes=["fallback profile json_recon"],
        ),
        ArtifactDraft(
            artifact_id="json-support-text",
            path=support_path,
            kind="plain_text",
            content_model={"lines": ["service endpoint: {}".format(support_path), "status: ok"]},
            review_notes=["fallback profile json_recon"],
        ),
    ]


def _build_plain_text_fallback_drafts(request: GenerationRequest) -> list[ArtifactDraft]:
    primary_path = request.normalized_path
    support_path = "/version" if primary_path != "/version" else "/info"
    return [
        ArtifactDraft(
            artifact_id="text-primary",
            path=primary_path,
            kind="plain_text",
            content_model={"lines": ["service endpoint: {}".format(primary_path), "status: ok"]},
            review_notes=["fallback profile text_recon"],
        ),
        ArtifactDraft(
            artifact_id="text-support",
            path=support_path,
            kind="plain_text",
            content_model={"lines": ["service endpoint: {}".format(support_path), "status: ok"]},
            review_notes=["fallback profile text_recon"],
        ),
    ]


def _build_binary_asset_fallback_drafts(request: GenerationRequest) -> list[ArtifactDraft]:
    primary_path = request.normalized_path
    return [
        ArtifactDraft(
            artifact_id="binary-primary",
            path=primary_path,
            kind="binary_asset",
            content_model={
                "content_type": _binary_asset_content_type_for_path(primary_path),
                "content_base64": _binary_asset_stub_base64(primary_path),
            },
            review_notes=["fallback profile binary_recon"],
        ),
        ArtifactDraft(
            artifact_id="binary-index",
            path="/index.html",
            kind="html_page",
            content_model={
                "title": "Asset Index",
                "heading": "Asset Index",
                "paragraphs": ["Static asset endpoint available."],
                "nav_links": [{"label": "Primary asset", "href": primary_path}],
                "images": [{"src": primary_path, "alt": "asset"}],
                "linked_stylesheets": [],
                "linked_scripts": [],
                "form": None,
                "footer": "Static asset preview",
            },
            review_notes=["fallback profile binary_recon"],
        ),
    ]


def _build_robots_fallback_drafts(request: GenerationRequest) -> list[ArtifactDraft]:
    return [
        ArtifactDraft(
            artifact_id="robots-primary",
            path=request.normalized_path,
            kind="robots_txt",
            content_model={"lines": ["User-agent: *", "Disallow: /private", "Disallow: /admin"]},
            review_notes=["fallback profile robots_recon"],
        ),
        ArtifactDraft(
            artifact_id="robots-index",
            path="/index.html",
            kind="html_page",
            content_model={
                "title": "Service Index",
                "heading": "Service Index",
                "paragraphs": ["Crawler policy is available."],
                "nav_links": [{"label": "robots", "href": request.normalized_path}],
                "images": [],
                "linked_stylesheets": [],
                "linked_scripts": [],
                "form": None,
                "footer": "Service resources",
            },
            review_notes=["fallback profile robots_recon"],
        ),
    ]


def _build_sitemap_fallback_drafts(request: GenerationRequest) -> list[ArtifactDraft]:
    primary_path = request.normalized_path
    return [
        ArtifactDraft(
            artifact_id="sitemap-primary",
            path=primary_path,
            kind="sitemap_xml",
            content_model={"urls": [primary_path, "/robots.txt", "/index.html"]},
            review_notes=["fallback profile sitemap_recon"],
        ),
        ArtifactDraft(
            artifact_id="sitemap-robots",
            path="/robots.txt",
            kind="robots_txt",
            content_model={"lines": ["User-agent: *", "Disallow: /private", "Disallow: /admin"]},
            review_notes=["fallback profile sitemap_recon"],
        ),
        ArtifactDraft(
            artifact_id="sitemap-index",
            path="/index.html",
            kind="html_page",
            content_model={
                "title": "Service Index",
                "heading": "Service Index",
                "paragraphs": ["Sitemap endpoint is published."],
                "nav_links": [
                    {"label": "sitemap", "href": primary_path},
                    {"label": "robots", "href": "/robots.txt"},
                ],
                "images": [],
                "linked_stylesheets": [],
                "linked_scripts": [],
                "form": None,
                "footer": "Service resources",
            },
            review_notes=["fallback profile sitemap_recon"],
        ),
    ]

def build_fallback_bundle(
    request: GenerationRequest,
    expert_spec: ExpertSpec | None = None,
    reasons: list[str] | None = None,
    max_artifacts: int | None = None,
 ) -> GeneratedBundle:
    intent_family = expert_spec.intent_family if expert_spec is not None else infer_intent_family(request.normalized_path)
    required_kind = _required_kind_for_path(request.normalized_path)
    if required_kind == "xml_document":
        profile_name = "xml_recon"
        drafts = _build_xml_fallback_drafts(request)
    elif required_kind == "json_document":
        profile_name = "json_recon"
        drafts = _build_json_fallback_drafts(request)
    elif required_kind == "plain_text":
        profile_name = "text_recon"
        drafts = _build_plain_text_fallback_drafts(request)
    elif required_kind == "binary_asset":
        profile_name = "binary_recon"
        drafts = _build_binary_asset_fallback_drafts(request)
    elif required_kind == "robots_txt":
        profile_name = "robots_recon"
        drafts = _build_robots_fallback_drafts(request)
    elif required_kind == "sitemap_xml":
        profile_name = "sitemap_recon"
        drafts = _build_sitemap_fallback_drafts(request)
    else:
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
