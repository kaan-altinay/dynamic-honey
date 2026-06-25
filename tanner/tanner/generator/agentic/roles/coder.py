from __future__ import annotations

import base64
import json
import re

from langgraph.types import Send

from tanner.generator.agentic.models import (
    ArtifactDraft,
    ExpertSpec,
    GenerationRequest,
    PlannedArtifact,
    ReferencePack,
    ResourcePlan,
    StructuredBackupManifestDraft,
    StructuredBinaryAssetDraft,
    StructuredConfigTextDraft,
    StructuredCredentialBaitDraft,
    StructuredHtmlPageDraft,
    StructuredJavascriptDraft,
    StructuredJsonDocumentDraft,
    StructuredLogExcerptDraft,
    StructuredPlainTextDraft,
    StructuredRobotsTxtDraft,
    StructuredSitemapDraft,
    StructuredStylesheetDraft,
    StructuredXmlDocumentDraft,
)
from tanner.generator.agentic.validators import (
    ValidationError,
    _is_external_reference,
    _normalize_allowed_paths,
    normalize_path,
    validate_artifact_draft,
    validate_artifact_draft_contract,
)
from tanner.generator.agentic.state import GraphState


class CoderRoleMixin:
    """Coder role: drafts and sanitizes the content of one planned artifact."""

    _PLACEHOLDER_TOKEN_RE = re.compile(r"\b(example|sample)\b", re.IGNORECASE)

    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

    async def _coder_node(self, state: GraphState):
        request = state["request"]
        expert_spec = state["expert_spec"]
        resource_plan = state["resource_plan"]
        artifact = state["pending_artifact"]
        plan_revision = state.get("plan_revision", 0)

        heuristic_draft = self._heuristic_draft(request, expert_spec, resource_plan, artifact, plan_revision)
        content_model_skeleton = json.dumps(
            self._content_model_skeleton_for_kind(artifact.kind),
            sort_keys=True,
        )
        draft_schema = self._coder_schema_for_kind(artifact.kind)
        reference_pack = state.get("reference_pack") or ReferencePack()
        artifact_reference_context = self._reference_context_for_artifact(reference_pack, artifact.artifact_id)
        local_asset_paths = artifact_reference_context.local_asset_paths if artifact_reference_context is not None else []
        allowed_local_asset_paths = (
            artifact_reference_context.allowed_local_asset_paths
            if artifact_reference_context is not None and artifact_reference_context.allowed_local_asset_paths
            else local_asset_paths
        )
        allowed_internal_paths = (
            artifact_reference_context.allowed_internal_paths
            if artifact_reference_context is not None and artifact_reference_context.allowed_internal_paths
            else [planned.path for planned in resource_plan.artifacts]
        )
        primary_path = artifact_reference_context.primary_path if artifact_reference_context is not None else resource_plan.primary_path
        forbidden_external_assets = (
            artifact_reference_context.forbidden_external_assets if artifact_reference_context is not None else True
        )
        reference_urls = artifact_reference_context.reference_urls if artifact_reference_context is not None else []
        reference_notes = artifact_reference_context.notes if artifact_reference_context is not None else []
        allowed_paths = sorted(set(allowed_local_asset_paths + allowed_internal_paths))
        coherence_facts = json.dumps(resource_plan.coherence_facts, sort_keys=True) if resource_plan.coherence_facts else "none"
        flow_metadata = "none"
        if artifact.flow_match_path or artifact.flow_condition is not None or artifact.flow_response is not None:
            flow_metadata = json.dumps(
                {
                    "flow_match_path": artifact.flow_match_path,
                    "flow_condition": artifact.flow_condition.model_dump(mode="json") if artifact.flow_condition is not None else None,
                    "flow_response": artifact.flow_response.model_dump(mode="json") if artifact.flow_response is not None else None,
                    "flow_priority": artifact.flow_priority,
                },
                sort_keys=True,
            )
        enable_scripted_flows = self.runtime_config.enable_scripted_flows
        messages = [
            {
                "role": "system",
                "content": self.prompts.coder.system(enable_scripted_flows),
            },
            {
                "role": "user",
                "content": self.prompts.coder.user(enable_scripted_flows).format(
                    artifact_id=artifact.artifact_id,
                    artifact_path=artifact.path,
                    kind=artifact.kind,
                    purpose=artifact.purpose,
                    response_status_code=artifact.response_contract.status_code,
                    response_content_type=artifact.response_contract.content_type
                    or self._default_content_type_for_artifact(artifact.kind, artifact.path),
                    response_headers_hint=", ".join(
                        "{}={}".format(header_hint.name, header_hint.value)
                        for header_hint in artifact.response_contract.headers_hint
                    )
                    if artifact.response_contract.headers_hint
                    else "none",
                    theme=expert_spec.environment_theme,
                    coherence_facts=coherence_facts,
                    flow_metadata=flow_metadata,
                    reference_urls=", ".join(reference_urls) if reference_urls else "none",
                    reference_notes=" | ".join(reference_notes) if reference_notes else "none",
                    must_reference=", ".join(artifact.links_to) if artifact.links_to else "none",
                    may_reference=", ".join([p for p in allowed_paths if p not in artifact.links_to]) if allowed_paths else "none",
                    kind_directive=self._kind_specific_directives(artifact.kind),
                    content_model_skeleton=content_model_skeleton,
                ),
            },
        ]

        try:
            structured_draft = await self._invoke_structured("coder", draft_schema, messages)
            structured_draft = draft_schema.model_validate(structured_draft)
            draft = self._materialize_structured_draft(structured_draft, artifact, plan_revision)
            draft = self._sanitize_artifact_draft(
                draft,
                request,
                allowed_local_asset_paths=allowed_local_asset_paths,
                allowed_internal_paths=allowed_internal_paths,
                primary_path=primary_path,
                forbidden_external_assets=forbidden_external_assets,
            )
            validate_artifact_draft(draft, request)
            try:
                validate_artifact_draft_contract(
                    draft,
                    request,
                    allowed_local_asset_paths=allowed_local_asset_paths,
                    allowed_internal_paths=allowed_internal_paths,
                    primary_path=primary_path,
                    forbidden_external_assets=forbidden_external_assets,
                )
            except ValidationError as contract_error:
                self.logger.info("Sanitizing coder draft for %s: %s", artifact.path, contract_error)
                draft = self._sanitize_artifact_draft(
                    draft,
                    request,
                    allowed_local_asset_paths=allowed_local_asset_paths,
                    allowed_internal_paths=allowed_internal_paths,
                    primary_path=primary_path,
                    forbidden_external_assets=forbidden_external_assets,
                )
                validate_artifact_draft(draft, request)
                validate_artifact_draft_contract(
                    draft,
                    request,
                    allowed_local_asset_paths=allowed_local_asset_paths,
                    allowed_internal_paths=allowed_internal_paths,
                    primary_path=primary_path,
                    forbidden_external_assets=forbidden_external_assets,
                )
        except Exception as error:
            self.logger.info("Falling back to heuristic draft for %s: %s", artifact.path, error)
            draft = self._sanitize_artifact_draft(
                heuristic_draft,
                request,
                allowed_local_asset_paths=allowed_local_asset_paths,
                allowed_internal_paths=allowed_internal_paths,
                primary_path=primary_path,
                forbidden_external_assets=forbidden_external_assets,
            )
            validate_artifact_draft(draft, request)
            validate_artifact_draft_contract(
                draft,
                request,
                allowed_local_asset_paths=allowed_local_asset_paths,
                allowed_internal_paths=allowed_internal_paths,
                primary_path=primary_path,
                forbidden_external_assets=forbidden_external_assets,
            )
            return {
                "artifact_drafts": [draft],
                "trace_notes": ["coder:heuristic:{}".format(draft.path)],
                "errors": ["coder fallback for {}: {} {}".format(artifact.path, error.__class__.__name__, error)],
                "generation_diagnostics": [
                    self._diagnostic_event(
                        "coder", "heuristic_fallback", artifact.path,
                        str(error), exception_type=error.__class__.__name__,
                        artifact_kind=artifact.kind,
                    )
                ],
            }

        return {
            "artifact_drafts": [draft],
            "trace_notes": ["coder:{}".format(draft.path)],
        }

    def _fan_out_coders(self, state: GraphState):
        request = state["request"]
        expert_spec = state["expert_spec"]
        resource_plan = state["resource_plan"]
        reference_pack = state.get("reference_pack")
        plan_revision = state.get("plan_revision", 0)
        sends = []
        for artifact in resource_plan.artifacts:
            sends.append(
                Send(
                    "coder_node",
                    {
                        "request": request,
                        "expert_spec": expert_spec,
                        "resource_plan": resource_plan,
                        "reference_pack": reference_pack,
                        "pending_artifact": artifact,
                        "plan_revision": plan_revision,
                    },
                )
            )
        return sends

    @staticmethod
    def _header_hints_to_dicts(header_hints):
        return [{header_hint.name: header_hint.value} for header_hint in header_hints]

    @staticmethod
    def _coder_schema_for_kind(kind: str):
        schema_by_kind = {
            "html_page": StructuredHtmlPageDraft,
            "config_text": StructuredConfigTextDraft,
            "json_document": StructuredJsonDocumentDraft,
            "plain_text": StructuredPlainTextDraft,
            "binary_asset": StructuredBinaryAssetDraft,
            "stylesheet": StructuredStylesheetDraft,
            "javascript": StructuredJavascriptDraft,
            "robots_txt": StructuredRobotsTxtDraft,
            "sitemap_xml": StructuredSitemapDraft,
            "xml_document": StructuredXmlDocumentDraft,
            "credential_bait": StructuredCredentialBaitDraft,
            "log_excerpt": StructuredLogExcerptDraft,
            "backup_manifest": StructuredBackupManifestDraft,
        }
        try:
            return schema_by_kind[kind]
        except KeyError as error:
            raise ValueError("Unsupported coder schema for kind {}".format(kind)) from error

    def _normalize_structured_content_model(self, kind: str, content_model) -> dict[str, object]:
        raw_content_model = content_model.model_dump(exclude_none=True)
        if kind == "stylesheet":
            for rule in raw_content_model.get("rules", []):
                rule["declarations"] = {
                    declaration["property"]: declaration["value"]
                    for declaration in rule.get("declarations", [])
                }
        if kind == "config_text" and raw_content_model.get("format") == "dotenv":
            raw_content_model["format"] = "env"
        return raw_content_model

    def _materialize_structured_draft(
        self,
        structured_draft,
        artifact: PlannedArtifact,
        plan_revision: int,
    ) -> ArtifactDraft:
        review_notes = list(structured_draft.review_notes)
        if structured_draft.path != artifact.path:
            review_notes.append(
                "normalized structured path {} -> {}".format(structured_draft.path, artifact.path)
            )
        if structured_draft.artifact_id != artifact.artifact_id:
            review_notes.append(
                "normalized structured artifact_id {} -> {}".format(
                    structured_draft.artifact_id,
                    artifact.artifact_id,
                )
            )

        contract_headers = self._header_hints_to_dicts(artifact.response_contract.headers_hint)
        structured_headers = self._header_hints_to_dicts(structured_draft.headers_hint)
        headers_hint = contract_headers + [
            header
            for header in structured_headers
            if header not in contract_headers
        ]

        return ArtifactDraft(
            artifact_id=artifact.artifact_id,
            path=artifact.path,
            kind=artifact.kind,
            content_model=self._normalize_structured_content_model(
                artifact.kind,
                structured_draft.content_model,
            ),
            status_code=artifact.response_contract.status_code,
            content_type=artifact.response_contract.content_type or structured_draft.content_type,
            headers_hint=headers_hint,
            review_notes=review_notes,
            plan_revision=plan_revision,
        )

    @staticmethod
    def _content_model_skeleton_for_kind(kind: str) -> dict[str, object]:
        skeletons: dict[str, dict[str, object]] = {
            "html_page": {
                "title": "<page title>",
                "heading": "<primary heading>",
                "paragraphs": ["<supporting paragraph>"],
                "nav_links": [{"label": "<nav label>", "href": "/<internal-path>"}],
                "images": [{"src": "/<local-asset-path>", "alt": "<image alt text>", "href": "/<optional-link-target>", "class_name": "<optional-class>"}],
                "linked_stylesheets": ["/<stylesheet-path>.css"],
                "linked_scripts": ["/<script-path>.js"],
                "form": {
                    "action": "/<submit-path>",
                    "method": "post",
                    "fields": [{"name": "<field-name>", "label": "<field label>", "type": "text"}],
                    "submit_label": "<submit label>",
                },
                "footer": "<footer text>",
            },
            "config_text": {
                "format": "<env-or-php-format>",
                "comment": "<configuration comment>",
                "entries": [{"key": "<config-key>", "value": "<config-value>"}],
            },
            "json_document": {
                "document": {"status": "ok", "version": "<value>", "items": []},
            },
            "plain_text": {
                "lines": ["<line one>", "<line two>"],
            },
            "binary_asset": {
                "content_type": "<mime-type>",
                "content_base64": "<base64-encoded-bytes>",
            },
            "stylesheet": {
                "rules": [{"selector": "<css selector>", "declarations": [{"property": "<property>", "value": "<value>"}]}],
            },
            "javascript": {
                "lines": ["<javascript line>", "<javascript line>"],
            },
            "robots_txt": {
                "lines": ["User-agent: *", "Disallow: /<path>"],
            },
            "sitemap_xml": {
                "urls": ["/<path>", "/<related-path>"],
            },
            "xml_document": {
                "lines": [
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                    "<root>",
                    "  <item key=\"example\">value</item>",
                    "</root>",
                ],
            },
            "log_excerpt": {
                "lines": ["[YYYY-MM-DD hh:mm:ss] level.MESSAGE: <log line>"],
            },
            "backup_manifest": {
                "lines": ["manifest-version: <number>", "include: /<path>"],
            },
            "credential_bait": {
                "lines": ["<credential-looking line>"],
            },
        }
        return skeletons.get(kind, {"lines": ["<artifact content line>"]})

    @staticmethod
    def _replace_placeholder_tokens(value: str) -> tuple[str, int]:
        def _replacement(match: re.Match[str]) -> str:
            token = match.group(0)
            if token.isupper():
                return "PRODUCTION"
            if token[:1].isupper():
                return "Production"
            return "production"

        sanitized_value, replacements = CoderRoleMixin._PLACEHOLDER_TOKEN_RE.subn(_replacement, value)
        return sanitized_value, replacements

    @staticmethod
    def _strip_control_chars(value: str) -> tuple[str, int]:
        sanitized_value, replacements = CoderRoleMixin._CONTROL_CHAR_RE.subn("", value)
        return sanitized_value, replacements

    @staticmethod
    def _scrub_control_chars(value):
        if isinstance(value, dict):
            total = 0
            sanitized = {}
            for key, item in value.items():
                sanitized_item, count = CoderRoleMixin._scrub_control_chars(item)
                sanitized[key] = sanitized_item
                total += count
            return sanitized, total
        if isinstance(value, list):
            total = 0
            sanitized = []
            for item in value:
                sanitized_item, count = CoderRoleMixin._scrub_control_chars(item)
                sanitized.append(sanitized_item)
                total += count
            return sanitized, total
        if isinstance(value, str):
            return CoderRoleMixin._strip_control_chars(value)
        return value, 0

    @staticmethod
    def _scrub_placeholder_content(value, *, parent_key: str | None = None):
        path_like_keys = {"path", "href", "src", "action", "local_path", "source_url"}
        if isinstance(value, dict):
            total = 0
            sanitized = {}
            for key, item in value.items():
                key_name = key if isinstance(key, str) else None
                sanitized_item, count = CoderRoleMixin._scrub_placeholder_content(item, parent_key=key_name)
                sanitized[key] = sanitized_item
                total += count
            return sanitized, total
        if isinstance(value, list):
            total = 0
            sanitized = []
            for item in value:
                sanitized_item, count = CoderRoleMixin._scrub_placeholder_content(item, parent_key=parent_key)
                sanitized.append(sanitized_item)
                total += count
            return sanitized, total
        if isinstance(value, str):
            stripped = value.strip()
            if parent_key in path_like_keys or stripped.startswith("/") or "://" in stripped:
                return value, 0
            return CoderRoleMixin._replace_placeholder_tokens(value)
        return value, 0

    @staticmethod
    def _sanitize_artifact_draft(
        draft: ArtifactDraft,
        request: GenerationRequest,
        *,
        allowed_local_asset_paths: list[str],
        allowed_internal_paths: list[str],
        primary_path: str,
        forbidden_external_assets: bool,
    ) -> ArtifactDraft:
        """Remove references that violate the contract instead of discarding the whole draft."""
        allowed_local = _normalize_allowed_paths(allowed_local_asset_paths, request)
        allowed_internal = _normalize_allowed_paths(allowed_internal_paths, request)
        allowed_internal.add(normalize_path(primary_path, index_page=request.index_page))
        allowed_paths = allowed_local | allowed_internal

        def _ref_allowed(ref: str, paths: set[str]) -> bool:
            if not isinstance(ref, str) or not ref.strip():
                return False
            if _is_external_reference(ref.strip()):
                return not forbidden_external_assets
            return normalize_path(ref.strip(), index_page=request.index_page) in paths

        model = dict(draft.content_model)
        notes = list(draft.review_notes)

        if draft.kind == "html_page":
            orig_nav = model.get("nav_links", [])
            filtered_nav = [
                link for link in orig_nav
                if isinstance(link, dict) and _ref_allowed(link.get("href", ""), allowed_internal)
            ]
            if len(filtered_nav) != len(orig_nav):
                notes.append("sanitized nav_links: removed {} invalid link(s)".format(len(orig_nav) - len(filtered_nav)))
                model["nav_links"] = filtered_nav

            orig_css = model.get("linked_stylesheets", [])
            filtered_css = [s for s in orig_css if isinstance(s, str) and _ref_allowed(s, allowed_paths)]
            if len(filtered_css) != len(orig_css):
                notes.append("sanitized linked_stylesheets: removed {} invalid ref(s)".format(len(orig_css) - len(filtered_css)))
                model["linked_stylesheets"] = filtered_css

            orig_js = model.get("linked_scripts", [])
            filtered_js = [s for s in orig_js if isinstance(s, str) and _ref_allowed(s, allowed_paths)]
            if len(filtered_js) != len(orig_js):
                notes.append("sanitized linked_scripts: removed {} invalid ref(s)".format(len(orig_js) - len(filtered_js)))
                model["linked_scripts"] = filtered_js

            orig_images = model.get("images", [])
            sanitized_images = []
            images_changed = False
            for img in orig_images:
                if not isinstance(img, dict):
                    continue
                if not _ref_allowed(img.get("src", ""), allowed_paths):
                    images_changed = True
                    continue
                href = img.get("href")
                if isinstance(href, str) and href.strip() and not _ref_allowed(href, allowed_internal):
                    img = {k: v for k, v in img.items() if k != "href"}
                    images_changed = True
                sanitized_images.append(img)
            if images_changed:
                notes.append("sanitized images: adjusted image references")
                model["images"] = sanitized_images

            form = model.get("form")
            if isinstance(form, dict) and isinstance(form.get("action"), str):
                if not _ref_allowed(form["action"], allowed_internal):
                    model["form"] = {**form, "action": draft.path}
                    notes.append("sanitized form.action -> {}".format(draft.path))

        model, control_char_replacements = CoderRoleMixin._scrub_control_chars(model)
        if control_char_replacements > 0:
            notes.append(
                "sanitized control characters: removed {} non-printable byte(s)".format(
                    control_char_replacements
                )
            )

        model, placeholder_replacements = CoderRoleMixin._scrub_placeholder_content(model)
        if placeholder_replacements > 0:
            notes.append(
                "sanitized placeholder text: replaced {} example/sample token(s)".format(
                    placeholder_replacements
                )
            )
        return ArtifactDraft(
            artifact_id=draft.artifact_id,
            path=draft.path,
            kind=draft.kind,
            content_model=model,
            status_code=draft.status_code,
            content_type=draft.content_type,
            headers_hint=draft.headers_hint,
            review_notes=notes,
            plan_revision=draft.plan_revision,
        )

    @staticmethod
    def _kind_specific_directives(kind: str) -> str:
        """Return kind-specific generation directives for the Coder role."""
        directives = {
            "html_page": (
                "For html_page: Include a form if the path suggests authentication (login, admin, signin). "
                "Use realistic form field names (username/password, not field1/field2). "
                "CSS classes should match common CMS conventions (e.g., WordPress uses 'login', 'button-primary'). "
                "Include nav_links to other artifacts in the bundle. "
                "Use environment_theme to inform the page tone and footer text."
            ),
            "config_text": (
                "For config_text: Generate key/value configuration text only for env/php style files. "
                "Use realistic infrastructure credentials with plausible hostnames and varied password patterns. "
                "Do not use config_text for XML, JSON, plain text, or binary payloads. "
                "Use format env for .env-like files and php for .php files."
            ),
            "json_document": (
                "For json_document: Output a realistic machine-readable JSON object or array in content_model.document. "
                "Use an array when the real API legitimately returns a top-level list (e.g. Docker /containers/json). "
                "Do not embed JSON inside string values and do not wrap JSON in prose or markdown fences."
            ),
            "plain_text": (
                "For plain_text: Output line-oriented text in content_model.lines with no HTML/XML/JSON wrappers. "
                "Use terse service-like responses suitable for text endpoints and probes."
            ),
            "binary_asset": (
                "For binary_asset: Set content_model.content_type to a concrete MIME type and provide content_model.content_base64 "
                "as valid base64 bytes for the file. Do not include prose or placeholders in base64 data."
            ),
            "stylesheet": (
                "For stylesheet: Use CSS selectors that match the environment_theme and linked HTML artifacts. "
                "Include realistic color schemes appropriate to the theme (e.g., WordPress blue #2271b1 for cms_probe). "
                "Define styles for common elements: body, forms, buttons, inputs."
            ),
            "javascript": (
                "For javascript: Keep scripts minimal and functional. "
                "Common patterns: form validation, field focus, simple DOM manipulation. "
                "Avoid complex logic that would require server-side state."
            ),
            "robots_txt": (
                "For robots_txt: Include realistic Disallow entries for admin, private, or backup paths. "
                "Use standard User-agent declarations."
            ),
            "sitemap_xml": (
                "For sitemap_xml: List all artifacts in the bundle plus the primary_path. "
                "Use absolute paths starting with /."
            ),
            "xml_document": (
                "For xml_document: Emit valid XML only. Start with an XML declaration and produce well-formed nested tags. "
                "Do not include env-style KEY=VALUE lines, markdown fences, or explanatory prose. "
                "Keep references as XML element values (e.g., URLs in tags) rather than config entries."
            ),
        }
        return directives.get(kind, "Generate realistic content appropriate to the artifact kind.")

    def _heuristic_draft(
        self,
        request: GenerationRequest,
        expert_spec: ExpertSpec,
        resource_plan: ResourcePlan,
        artifact: PlannedArtifact,
        plan_revision: int,
    ) -> ArtifactDraft:
        linked_stylesheets = [link for link in artifact.links_to if link.endswith(".css")]
        linked_scripts = [link for link in artifact.links_to if link.endswith(".js")]
        nav_links = [
            {"label": self._nav_label(link), "href": link}
            for link in artifact.links_to
            if not link.endswith(".css") and not link.endswith(".js") and link != artifact.path
        ]

        if artifact.kind == "html_page":
            is_login_like = any(token in artifact.path.lower() for token in ["login", "admin", "wp-login"])
            content_model = {
                "title": self._page_title_for_artifact(artifact.path, expert_spec.environment_theme),
                "heading": self._page_heading_for_artifact(artifact.path),
                "paragraphs": self._heuristic_page_copy(expert_spec.intent_family),
                "linked_stylesheets": linked_stylesheets,
                "linked_scripts": linked_scripts,
                "nav_links": nav_links,
                "footer": expert_spec.environment_theme,
            }
            if is_login_like:
                content_model["form"] = {
                    "action": artifact.path,
                    "method": "post",
                    "fields": [
                        {"name": "username", "label": "Username", "type": "text"},
                        {"name": "password", "label": "Password", "type": "password"},
                    ],
                    "submit_label": "Sign In",
                }
        elif artifact.kind == "config_text":
            format_name = "php" if artifact.path.endswith(".php") else "env"
            content_model = {
                "format": format_name,
                "comment": self._config_comment_for_path(artifact.path),
                "entries": self._config_entries_for_path(artifact.path),
            }
        elif artifact.kind == "json_document":
            content_model = {
                "document": {
                    "status": "ok",
                    "path": artifact.path,
                    "service": expert_spec.environment_theme,
                    "timestamp": "2026-04-03T00:13:12Z",
                }
            }
        elif artifact.kind == "plain_text":
            content_model = {
                "lines": [
                    "service endpoint: {}".format(artifact.path),
                    "status: ok",
                ]
            }
        elif artifact.kind == "binary_asset":
            content_model = {
                "content_type": self._binary_asset_content_type_for_path(artifact.path),
                "content_base64": self._binary_asset_stub_base64(artifact.path),
            }
        elif artifact.kind == "stylesheet":
            content_model = {
                "rules": [
                    {"selector": "body", "declarations": {"font-family": "Arial, sans-serif", "background": "#f5f7fa", "color": "#1f2937"}},
                    {"selector": "main", "declarations": {"max-width": "720px", "margin": "4rem auto", "padding": "2rem", "background": "#ffffff", "border": "1px solid #d0d7de"}},
                    {"selector": "button", "declarations": {"background": "#2271b1", "color": "#ffffff", "border": "none", "padding": "0.75rem 1rem"}},
                ]
            }
        elif artifact.kind == "javascript":
            content_model = {
                "lines": [
                    "document.addEventListener('DOMContentLoaded', function () {",
                    "  var firstField = document.querySelector('input');",
                    "  if (firstField) { firstField.focus(); }",
                    "});",
                ]
            }
        elif artifact.kind == "robots_txt":
            content_model = {"lines": ["User-agent: *", "Disallow: /private", "Disallow: /admin"]}
        elif artifact.kind == "sitemap_xml":
            content_model = {"urls": [request.normalized_path] + [planned.path for planned in resource_plan.artifacts if planned.path != request.normalized_path]}
        elif artifact.kind == "xml_document":
            descriptor_name = artifact.path.rsplit("/", 1)[-1] or "descriptor.xml"
            content_model = {
                "lines": [
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                    "<root xmlns=\"urn:schemas-upnp-org:device-1-0\">",
                    "  <specVersion><major>1</major><minor>0</minor></specVersion>",
                    "  <URLBase>http://10.44.12.1:1900/</URLBase>",
                    "  <device>",
                    "    <friendlyName>Northbridge Branch Gateway</friendlyName>",
                    "    <modelName>{}</modelName>".format(descriptor_name),
                    "    <serviceList>",
                    "      <service>",
                    "        <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>",
                    "        <SCPDURL>/WANCfgSCPD.xml</SCPDURL>",
                    "      </service>",
                    "    </serviceList>",
                    "  </device>",
                    "</root>",
                ]
            }
        elif artifact.kind == "log_excerpt":
            content_model = {
                "lines": [
                    "[2026-04-03 00:13:12] production.INFO: request path {} inspected".format(request.normalized_path),
                    "[2026-04-03 00:13:15] production.WARNING: unexpected credential validation attempt detected",
                    "[2026-04-03 00:13:18] production.INFO: asset reconciliation completed",
                ]
            }
        elif artifact.kind == "backup_manifest":
            content_model = {
                "lines": [
                    "manifest-version: 1",
                    "snapshot: nightly-2026-04-03",
                    "include: {}".format(request.normalized_path),
                    "checksum-policy: md5",
                ]
            }
        else:
            content_model = {"lines": [artifact.purpose]}

        contract_headers = self._header_hints_to_dicts(artifact.response_contract.headers_hint)
        draft = ArtifactDraft(
            artifact_id=artifact.artifact_id,
            path=artifact.path,
            kind=artifact.kind,
            content_model=content_model,
            status_code=artifact.response_contract.status_code,
            content_type=artifact.response_contract.content_type
            or self._default_content_type_for_artifact(artifact.kind, artifact.path),
            headers_hint=contract_headers + [{"X-Tanner-Generated": "agentic"}],
            review_notes=[artifact.purpose],
            plan_revision=plan_revision,
        )
        validate_artifact_draft(draft, request)
        return draft

    @staticmethod
    def _page_title_for_artifact(path: str, theme: str) -> str:
        lowered = path.lower()
        if "wp-login" in lowered or "login" in lowered:
            return "WordPress Login" if "wp" in lowered else "Administrative Login"
        if "admin" in lowered:
            return "Administration Console"
        return theme

    @staticmethod
    def _page_heading_for_artifact(path: str) -> str:
        tail = path.rstrip("/").split("/")[-1] or "index"
        if tail.endswith(".php"):
            tail = tail[:-4]
        if tail.endswith(".html"):
            tail = tail[:-5]
        words = [segment.capitalize() for segment in tail.replace("-", " ").replace("_", " ").split()]
        return " ".join(words) or "Overview"

    @staticmethod
    def _nav_label(path: str) -> str:
        normalized = normalize_path(path)
        if normalized == "/wp-login.php":
            return "Lost Password"
        return CoderRoleMixin._page_heading_for_artifact(normalized)

    @staticmethod
    def _config_comment_for_path(path: str) -> str:
        if path.endswith("wp-config.php") or path.endswith(".php"):
            return "Application configuration"
        return "Application environment configuration"

    @staticmethod
    def _config_entries_for_path(path: str) -> list[dict[str, str]]:
        if path.endswith("wp-config.php"):
            return [
                {"key": "DB_NAME", "value": "wordpress_prod"},
                {"key": "DB_USER", "value": "wp_service"},
                {"key": "DB_PASSWORD", "value": "W0rdPress!2026"},
                {"key": "DB_HOST", "value": "10.24.18.21"},
                {"key": "AUTH_KEY", "value": "wordpress-auth-key-2026"},
                {"key": "SECURE_AUTH_KEY", "value": "wordpress-secure-auth-key-2026"},
                {"key": "LOGGED_IN_KEY", "value": "wordpress-logged-in-key-2026"},
                {"key": "NONCE_KEY", "value": "wordpress-nonce-key-2026"},
            ]
        return [
            {"key": "APP_NAME", "value": "customer-portal"},
            {"key": "APP_ENV", "value": "production"},
            {"key": "APP_KEY", "value": "base64:O0vJm0QW2N7PkQemv3a8sB2sS4oI1C4e"},
            {"key": "APP_URL", "value": "https://portal.example.internal"},
            {"key": "APP_DEBUG", "value": "false"},
            {"key": "LOG_CHANNEL", "value": "stack"},
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
        ]

    @staticmethod
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

    @staticmethod
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

    def _default_content_type_for_artifact(self, kind: str, path: str) -> str:
        if kind == "html_page":
            return "text/html; charset=utf-8"
        if kind == "config_text":
            return "text/plain; charset=utf-8"
        if kind == "json_document":
            return "application/json; charset=utf-8"
        if kind in {"plain_text", "robots_txt", "credential_bait", "log_excerpt", "backup_manifest"}:
            return "text/plain; charset=utf-8"
        if kind == "binary_asset":
            return self._binary_asset_content_type_for_path(path)
        if kind == "stylesheet":
            return "text/css; charset=utf-8"
        if kind == "javascript":
            return "application/javascript; charset=utf-8"
        if kind in {"sitemap_xml", "xml_document"}:
            return "application/xml; charset=utf-8"
        return "application/octet-stream"

    @staticmethod
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
