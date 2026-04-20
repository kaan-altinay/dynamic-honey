from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


IntentFamily = Literal[
    "config_theft",
    "admin_portal",
    "framework_probe",
    "backup_probe",
    "cms_probe",
    "generic_recon",
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
    depends_on: list[str] = Field(default_factory=list)
    links_to: list[str] = Field(default_factory=list)
    must_exist: bool = True
    render_strategy: str = "deterministic"
    artifact_scope: ArtifactScope = "static_file"
    dynamic_candidate: bool = False
    service_candidate: bool = False


class ResourcePlan(ModelBase):
    primary_path: str
    theme_summary: str
    artifacts: list[PlannedArtifact]
    reference_asset_plan: ReferenceAssetPlan = Field(default_factory=ReferenceAssetPlan)
    bundle_budget_count: int = Field(ge=1)
    bundle_budget_bytes: int = Field(ge=1)
    static_only: bool = True
    review_focus: list[str] = Field(default_factory=list)


class HeaderHint(ModelBase):
    name: str
    value: str


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
    document: dict[str, Any] = Field(default_factory=dict)


class BinaryAssetContent(ModelBase):
    content_type: str = "application/octet-stream"
    content_base64: str



class StructuredDraftBase(ModelBase):
    artifact_id: str
    path: str
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


class GeneratedBundle(ModelBase):
    primary_path: str
    artifacts: list[GeneratedArtifact]
    review_summary: str
    used_fallback: bool = False


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
    max_review_loops: int = Field(default=2, ge=1)
    max_bundle_artifacts: int = Field(default=4, ge=1)
    max_bundle_bytes: int = Field(default=262_144, ge=1024)
    checkpoint_path: str = "/tmp/tanner-agentic-checkpoints.sqlite"
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
