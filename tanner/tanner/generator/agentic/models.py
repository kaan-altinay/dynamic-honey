from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


IntentFamily = Literal[
    "config_theft",
    "admin_portal",
    "framework_probe",
    "backup_probe",
    "cms_probe",
    "generic_recon",
    "auth_portal",
    "network_device",
    "container_api",
    "kubernetes_api",
    "elastic_api",
    "solr_api",
    "config_secret",
    "webshell_probe",
]

ArtifactKind = Literal[
    "html_page",
    "config_text",
    "json_document",
    "plain_text",
    "binary_asset",
    "stylesheet",
    "javascript",
    "robots_txt",
    "sitemap_xml",
    "xml_document",
    "credential_bait",
    "log_excerpt",
    "backup_manifest",
    "asset_file",
]

AssetFetchKind = Literal["image", "stylesheet", "script", "icon", "font", "other"]
ArtifactScope = Literal["static_file", "dynamic_endpoint", "service_stub"]
RoleName = Literal["expert", "design", "coder", "review"]


class ModelBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationRequest(ModelBase):
    host: str | None = None
    requested_path: str
    normalized_path: str
    site_profile: dict[str, Any] = Field(default_factory=dict)
    request_kind: Literal["seed", "runtime_miss"] = "runtime_miss"
    index_page: str = "/index.html"

class EndpointSemanticHint(ModelBase):
    """General V2 hint derived from path terms; advisory prompt context only."""
    path_tokens: list[str] = Field(default_factory=list)
    likely_resource_terms: list[str] = Field(default_factory=list)
    likely_protocol_terms: list[str] = Field(default_factory=list)
    likely_interaction_styles: list[str] = Field(default_factory=list)
    suggested_response_shapes: list[str] = Field(default_factory=list)



class ExpertSpec(ModelBase):
    intent_family: IntentFamily
    attacker_goal: str
    confidence: float = Field(ge=0.0, le=1.0)
    primary_resource_kind: ArtifactKind
    lure_requirements: list[str] = Field(default_factory=list)
    supporting_context: list[str] = Field(default_factory=list)
    environment_theme: str
    references: list[str] = Field(default_factory=list)


class PlannedAssetFetch(ModelBase):
    asset_id: str
    source_url: str
    local_path: str
    kind: AssetFetchKind
    required_for_artifact_ids: list[str] = Field(default_factory=list)
    reason: str


class ReferenceAssetPlan(ModelBase):
    reference_urls: list[str] = Field(default_factory=list)
    asset_fetches: list[PlannedAssetFetch] = Field(default_factory=list)


class PlannedArtifact(ModelBase):
    artifact_id: str
    path: str
    kind: ArtifactKind
    purpose: str
    response_contract: ResponseContract = Field(default_factory=lambda: ResponseContract())
    depends_on: list[str] = Field(default_factory=list)
    links_to: list[str] = Field(default_factory=list)
    must_exist: bool = True
    render_strategy: str = "deterministic"
    artifact_scope: ArtifactScope = "static_file"
    dynamic_candidate: bool = False
    service_candidate: bool = False
    flow_match_path: str | None = None
    flow_condition: FlowCondition | None = None
    flow_response: FlowResponse | None = None
    flow_priority: int = 0


class ResourcePlan(ModelBase):
    primary_path: str
    theme_summary: str
    artifacts: list[PlannedArtifact]
    reference_asset_plan: ReferenceAssetPlan = Field(default_factory=ReferenceAssetPlan)
    bundle_budget_count: int = Field(ge=1)
    bundle_budget_bytes: int = Field(ge=1)
    static_only: bool = True
    review_focus: list[str] = Field(default_factory=list)
    coherence_facts: dict[str, Any] = Field(default_factory=dict)


class HeaderHint(ModelBase):
    name: str
    value: str


class ResponseContract(ModelBase):
    status_code: int = Field(default=200, ge=100, le=599)
    content_type: str | None = None
    headers_hint: list[HeaderHint] = Field(default_factory=list)

class LinkSpec(ModelBase):
    label: str
    href: str


class ImageSpec(ModelBase):
    src: str
    alt: str
    href: str | None = None
    class_name: str | None = None


class FormFieldSpec(ModelBase):
    name: str
    label: str
    type: str


class FormSpec(ModelBase):
    action: str
    method: str
    fields: list[FormFieldSpec] = Field(default_factory=list)
    submit_label: str


class HtmlPageContent(ModelBase):
    title: str
    heading: str
    paragraphs: list[str] = Field(default_factory=list)
    nav_links: list[LinkSpec] = Field(default_factory=list)
    images: list[ImageSpec] = Field(default_factory=list)
    linked_stylesheets: list[str] = Field(default_factory=list)
    linked_scripts: list[str] = Field(default_factory=list)
    form: FormSpec | None = None
    footer: str = ""


class ConfigEntry(ModelBase):
    key: str
    value: str


class ConfigTextContent(ModelBase):
    format: Literal["env", "php", "dotenv"] = "env"
    comment: str | None = None
    entries: list[ConfigEntry] = Field(default_factory=list)


class CssDeclaration(ModelBase):
    property: str
    value: str


class StylesheetRule(ModelBase):
    selector: str
    declarations: list[CssDeclaration] = Field(default_factory=list)


class StylesheetContent(ModelBase):
    rules: list[StylesheetRule] = Field(default_factory=list)


class LineArtifactContent(ModelBase):
    lines: list[str] = Field(default_factory=list)


class SitemapContent(ModelBase):
    urls: list[str] = Field(default_factory=list)

class JsonDocumentContent(ModelBase):
    """Wraps an arbitrary JSON document.

    Real-world JSON APIs this models (Docker, Kubernetes, search engines, ...)
    legitimately return either a top-level object or a top-level array, and
    LLM-produced drafts sometimes place document fields as siblings of
    ``document`` instead of nesting them. Coalesce both cases before
    field validation rather than rejecting otherwise-correct content.
    """

    document: dict[str, Any] | list[Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coalesce_document(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"document": data}
        if isinstance(data, dict):
            nested = data.get("document")
            extra = {key: value for key, value in data.items() if key != "document"}
            if extra and isinstance(nested, dict):
                return {"document": {**extra, **nested}}
            if extra and nested is None:
                return {"document": extra}
        return data


class BinaryAssetContent(ModelBase):
    content_type: str = "application/octet-stream"
    content_base64: str



class StructuredDraftBase(ModelBase):
    artifact_id: str
    path: str
    status_code: int = Field(default=200, ge=100, le=599)
    content_type: str | None = None
    headers_hint: list[HeaderHint] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class StructuredHtmlPageDraft(StructuredDraftBase):
    kind: Literal["html_page"] = "html_page"
    content_model: HtmlPageContent


class StructuredConfigTextDraft(StructuredDraftBase):
    kind: Literal["config_text"] = "config_text"
    content_model: ConfigTextContent


class StructuredJsonDocumentDraft(StructuredDraftBase):
    kind: Literal["json_document"] = "json_document"
    content_model: JsonDocumentContent


class StructuredStylesheetDraft(StructuredDraftBase):
    kind: Literal["stylesheet"] = "stylesheet"
    content_model: StylesheetContent


class StructuredJavascriptDraft(StructuredDraftBase):
    kind: Literal["javascript"] = "javascript"
    content_model: LineArtifactContent


class StructuredRobotsTxtDraft(StructuredDraftBase):
    kind: Literal["robots_txt"] = "robots_txt"
    content_model: LineArtifactContent


class StructuredPlainTextDraft(StructuredDraftBase):
    kind: Literal["plain_text"] = "plain_text"
    content_model: LineArtifactContent


class StructuredSitemapDraft(StructuredDraftBase):
    kind: Literal["sitemap_xml"] = "sitemap_xml"
    content_model: SitemapContent


class StructuredCredentialBaitDraft(StructuredDraftBase):
    kind: Literal["credential_bait"] = "credential_bait"
    content_model: LineArtifactContent


class StructuredLogExcerptDraft(StructuredDraftBase):
    kind: Literal["log_excerpt"] = "log_excerpt"
    content_model: LineArtifactContent


class StructuredBackupManifestDraft(StructuredDraftBase):
    kind: Literal["backup_manifest"] = "backup_manifest"
    content_model: LineArtifactContent

class StructuredBinaryAssetDraft(StructuredDraftBase):
    kind: Literal["binary_asset"] = "binary_asset"
    content_model: BinaryAssetContent


class StructuredXmlDocumentDraft(StructuredDraftBase):
    kind: Literal["xml_document"] = "xml_document"
    content_model: LineArtifactContent



class ArtifactDraft(ModelBase):
    artifact_id: str
    path: str
    kind: ArtifactKind
    content_model: dict[str, Any] = Field(default_factory=dict)
    status_code: int = Field(default=200, ge=100, le=599)
    content_type: str | None = None
    headers_hint: list[dict[str, str]] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)
    plan_revision: int = Field(default=0, ge=0)


class AssetCandidate(ModelBase):
    source_url: str
    kind: AssetFetchKind
    tag: str
    local_path_hint: str
    note: str = ""


class ReferencePage(ModelBase):
    url: str
    final_url: str
    title: str = ""
    text_excerpt: str = ""
    asset_candidates: list[AssetCandidate] = Field(default_factory=list)


class FetchedAsset(ModelBase):
    asset_id: str
    source_url: str
    local_path: str
    kind: AssetFetchKind
    content_type: str
    body_bytes: bytes
    required_for_artifact_ids: list[str] = Field(default_factory=list)


class ArtifactReferenceContext(ModelBase):
    artifact_id: str
    reference_urls: list[str] = Field(default_factory=list)
    local_asset_paths: list[str] = Field(default_factory=list)
    allowed_local_asset_paths: list[str] = Field(default_factory=list)
    allowed_internal_paths: list[str] = Field(default_factory=list)
    primary_path: str = "/index.html"
    forbidden_external_assets: bool = True
    notes: list[str] = Field(default_factory=list)


class ReferencePack(ModelBase):
    reference_pages: list[ReferencePage] = Field(default_factory=list)
    fetched_assets: list[FetchedAsset] = Field(default_factory=list)
    artifact_contexts: list[ArtifactReferenceContext] = Field(default_factory=list)


class GeneratedArtifact(ModelBase):
    path: str
    kind: ArtifactKind
    headers: list[dict[str, str]] = Field(default_factory=list)
    body_bytes: bytes
    status_code: int = Field(default=200, ge=100, le=599)
    source_artifact_id: str
    artifact_scope: ArtifactScope = "static_file"




# ─── V2: Flow Descriptor Models ──────────────────────────────────────────────

class FlowCondition(ModelBase):
    """Conditions evaluated against live session state to decide if a rule fires.

    For credential/setup/admin forms, use method="POST" rules on the form action
    itself.  Cookie/history guards are for protected pages, not a substitute for
    POST failure handling.  When a form supports repeated invalid submissions,
    encode the cooldown explicitly with min_prior_post_count_to_path,
    lockout_window_seconds, and lockout_active.
    """
    requires_cookie: str | None = None            # cookie key must be present
    missing_cookie: str | None = None             # cookie key must be absent
    requires_prev_path: str | None = None         # last path in history must match
    method: str | None = None                     # HTTP method (GET / POST / ...)
    min_post_count_to_path: int | None = None     # floor for lockout simulation
    min_prior_post_count_to_path: int | None = None  # prior POSTs before current request
    lockout_window_seconds: int | None = None     # cooldown duration after threshold
    lockout_active: bool | None = None            # require active/inactive cooldown state
    requires_header: str | None = None            # header key must be present
    missing_header: str | None = None             # header key must be absent
    header_equals: dict[str, str] = Field(default_factory=dict)
    header_contains: dict[str, str] = Field(default_factory=dict)
    query_has: list[str] = Field(default_factory=list)
    query_equals: dict[str, str] = Field(default_factory=dict)
    query_contains: dict[str, str] = Field(default_factory=dict)
    post_has: list[str] = Field(default_factory=list)
    post_equals: dict[str, str] = Field(default_factory=dict)
    post_contains: dict[str, str] = Field(default_factory=dict)


class FlowResponse(ModelBase):
    """Action taken when a flow rule matches.

    artifact_path means rewrite and serve a concrete generated artifact.  For
    invalid credential submissions prefer an artifact_path that visibly contains
    failure feedback over a bare redirect back to the original form.  redirect_to
    is appropriate for protected-page guards, logout, and success transitions.
    set_cookie maps cookie-name -> cookie-value only; attributes such as
    HttpOnly/Path/SameSite/Secure are not separate keys here.
    """
    artifact_path: str | None = None              # rewrite: serve this meta key
    redirect_to: str | None = None                # synthetic 302 Location
    status_code: int = 200
    set_cookie: dict[str, str] = Field(default_factory=dict)
    clear_cookie: list[str] = Field(default_factory=list)
    headers: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("set_cookie", mode="before")
    @classmethod
    def _normalize_set_cookie(cls, value):
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            return value
        cookie_attribute_keys = {
            "httponly",
            "path",
            "secure",
            "samesite",
            "domain",
            "max-age",
            "expires",
        }
        normalized = {}
        for key, raw_value in value.items():
            if not isinstance(key, str) or not key.strip():
                continue
            if key.strip().lower() in cookie_attribute_keys:
                continue
            if isinstance(raw_value, str):
                normalized[key] = raw_value
            elif isinstance(raw_value, (int, float, bool)):
                normalized[key] = str(raw_value)
            elif raw_value is not None:
                normalized[key] = str(raw_value)
        return normalized


class FlowRule(ModelBase):
    """Single conditional routing rule; higher priority is evaluated first."""
    match_path: str
    condition: FlowCondition | None = None        # None = always matches this path
    response: FlowResponse
    priority: int = 0
    # Path of the /_flow/ artifact that defined this rule (first-class metadata
    # or variant-name convention). Distinct from response.artifact_path: a
    # redirect_to-only rule still has a source artifact even though it serves
    # no content of its own. Used by diagnose_flow_reachability to recognize
    # such artifacts as intentionally tied in rather than orphaned.
    source_artifact_path: str | None = None


class FlowDescriptor(ModelBase):
    """Complete scripted-flow specification for a generated V2 bundle.

    A credential/setup/admin form descriptor must be behaviorally executable,
    not just syntactically valid: every POST form action needs a POST-specific
    failure artifact for the first three invalid submissions, followed by a
    one-minute lockout artifact for later POSTs during the cooldown window.
    After the cooldown elapses, the same three-attempt loop begins again.
    Missing-cookie redirects alone are insufficient for form handlers because
    they hide the invalid-credential state from the client.
    """
    rules: list[FlowRule]

class GeneratedBundle(ModelBase):
    primary_path: str
    artifacts: list[GeneratedArtifact]
    review_summary: str
    used_fallback: bool = False
    flow_descriptor: FlowDescriptor | None = None
    generation_trace: list[str] = Field(default_factory=list)
    generation_errors: list[str] = Field(default_factory=list)
    generation_diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class StructuredReviewDecision(ModelBase):
    decision: Literal["approve", "revise", "fallback"]
    reasons: list[str] = Field(default_factory=list)
    required_fixes: list[str] = Field(default_factory=list)


class ReviewDecision(ModelBase):
    decision: Literal["approve", "revise", "fallback"]
    reasons: list[str] = Field(default_factory=list)
    required_fixes: list[str] = Field(default_factory=list)


class GeneratorRoleConfig(ModelBase):
    provider: str = "ollama"
    model: str = "qwen2.5:14b-instruct"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=900, ge=64)
    timeout: int = Field(default=45, ge=1)
    max_retries: int = Field(default=2, ge=0)


class GeneratorRuntimeConfig(ModelBase):
    backend: str = "agentic"
    enable_scripted_flows: bool = False
    max_review_loops: int = Field(default=2, ge=1)
    max_design_validation_loops: int = Field(default=2, ge=1)
    allow_fallback_persistence: bool = False
    max_bundle_artifacts: int = Field(default=4, ge=1)
    max_bundle_bytes: int = Field(default=262_144, ge=1024)
    checkpoint_path: str = "/tmp/tanner-agentic-checkpoints.sqlite"
    graph_recursion_limit: int = Field(default=200, ge=25)
    review_log_path: str = "/tmp/tanner-agentic-review-log.json"
    enable_live_research: bool = True
    max_tool_response_chars: int = Field(default=4_000, ge=256)
    max_command_output_chars: int = Field(default=4_000, ge=256)
    command_timeout: int = Field(default=5, ge=1)
    max_concurrent_model_calls: int = Field(default=4, ge=1)
    inter_call_delay_seconds: float = Field(default=0.0, ge=0.0)
    max_rate_limit_retries: int = Field(default=2, ge=0)
    default_rate_limit_backoff_seconds: float = Field(default=12.0, ge=0.0)
    max_length_limit_retries: int = Field(default=2, ge=0)
    length_retry_token_increase: int = Field(default=800, ge=1)
    max_length_retry_tokens: int = Field(default=6000, ge=64)
    roles: dict[RoleName, GeneratorRoleConfig]

    def role_config(self, role_name: RoleName) -> GeneratorRoleConfig:
        return self.roles[role_name]


class ResearchResult(ModelBase):
    query: str
    snippets: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class CommandResult(ModelBase):
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
