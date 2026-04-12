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
    "stylesheet",
    "javascript",
    "robots_txt",
    "sitemap_xml",
    "credential_bait",
    "log_excerpt",
    "backup_manifest",
]

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
    primary_resource_kind: str
    lure_requirements: list[str] = Field(default_factory=list)
    supporting_context: list[str] = Field(default_factory=list)
    environment_theme: str
    references: list[str] = Field(default_factory=list)


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
    bundle_budget_count: int = Field(ge=1)
    bundle_budget_bytes: int = Field(ge=1)
    static_only: bool = True
    review_focus: list[str] = Field(default_factory=list)


class HeaderHint(ModelBase):
    name: str
    value: str


class StructuredArtifactDraft(ModelBase):
    artifact_id: str
    path: str
    kind: ArtifactKind
    content_model_json: str = Field(min_length=2)
    headers_hint: list[HeaderHint] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class ArtifactDraft(ModelBase):
    artifact_id: str
    path: str
    kind: ArtifactKind
    content_model: dict[str, Any] = Field(default_factory=dict)
    headers_hint: list[dict[str, str]] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)
    plan_revision: int = Field(default=0, ge=0)

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
