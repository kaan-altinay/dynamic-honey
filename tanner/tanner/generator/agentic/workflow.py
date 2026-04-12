from __future__ import annotations

import asyncio
import json
import logging
import operator
import re
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from tanner.generator.agentic.config import load_runtime_config
from tanner.generator.agentic.fallback import build_fallback_bundle
from tanner.generator.agentic.model_factory import build_role_model
from tanner.generator.agentic.models import (
    ArtifactDraft,
    ExpertSpec,
    GeneratedBundle,
    GenerationRequest,
    GeneratorRuntimeConfig,
    PlannedArtifact,
    ResourcePlan,
    ReviewDecision,
    StructuredArtifactDraft,
 )
from tanner.generator.agentic.renderers import render_artifact
from tanner.generator.agentic.tools import inspect_reference, sandbox_command, web_research
from tanner.generator.agentic.validators import (
    ValidationError,
    ensure_generation_request,
    infer_intent_family,
    normalize_path,
    validate_artifact_draft,
    validate_bundle,
    validate_plan,
)
from tanner.generator.base_generator import BaseGenerator


class GraphState(TypedDict, total=False):
    request: GenerationRequest
    expert_spec: ExpertSpec
    resource_plan: ResourcePlan
    pending_artifact: PlannedArtifact
    artifact_drafts: Annotated[list[ArtifactDraft], operator.add]
    review_decision: ReviewDecision
    review_iteration: int
    review_feedback: list[str]
    generated_bundle: GeneratedBundle
    trace_notes: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    plan_revision: int


class AgenticBundleGenerator(BaseGenerator):
    def __init__(self, runtime_config: GeneratorRuntimeConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.runtime_config = runtime_config or load_runtime_config()
        self._role_models = {}
        self._invoke_semaphore = asyncio.Semaphore(self.runtime_config.max_concurrent_model_calls)
        self._invoke_spacing_lock = asyncio.Lock()
        self._next_model_call_time = 0.0
        self._graph_builder = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("normalize_request", self._normalize_request_node)
        builder.add_node("expert_node", self._expert_node)
        builder.add_node("design_node", self._design_node)
        builder.add_node("coder_node", self._coder_node)
        builder.add_node("assemble_bundle", self._assemble_bundle)
        builder.add_node("review_node", self._review_node)
        builder.add_node("finalize_bundle", self._finalize_bundle)
        builder.add_node("fallback_node", self._fallback_node)

        builder.add_edge(START, "normalize_request")
        builder.add_edge("normalize_request", "expert_node")
        builder.add_edge("expert_node", "design_node")
        builder.add_conditional_edges("design_node", self._fan_out_coders, ["coder_node"])
        builder.add_edge("coder_node", "assemble_bundle")
        builder.add_edge("assemble_bundle", "review_node")
        builder.add_conditional_edges(
            "review_node",
            self._route_after_review,
            {"approve": "finalize_bundle", "revise": "design_node", "fallback": "fallback_node"},
        )
        builder.add_edge("finalize_bundle", END)
        builder.add_edge("fallback_node", END)
        return builder

    def _get_role_model(self, role_name: str):
        if role_name in self._role_models:
            return self._role_models[role_name]

        try:
            model = build_role_model(role_name, self.runtime_config)
        except Exception as error:
            self.logger.warning("Unable to initialize %s model: %s", role_name, error)
            model = None
        self._role_models[role_name] = model
        return model

    def _build_role_model_for_attempt(self, role_name: str, max_tokens_override: int | None = None):
        if max_tokens_override is None:
            return self._get_role_model(role_name)
        return build_role_model(
            role_name,
            self.runtime_config,
            max_tokens_override=max_tokens_override,
        )

    @staticmethod
    def _is_length_limit_error(error: Exception) -> bool:
        if error.__class__.__name__ == "LengthFinishReasonError":
            return True
        message = str(error).lower()
        return (
            "length limit was reached" in message
            or "max completion tokens reached" in message
            or "json_validate_failed" in message and "valid document" in message
        )

    def _next_length_retry_tokens(self, current_max_tokens: int) -> int | None:
        next_max_tokens = min(
            current_max_tokens + self.runtime_config.length_retry_token_increase,
            self.runtime_config.max_length_retry_tokens,
        )
        if next_max_tokens <= current_max_tokens:
            return None
        return next_max_tokens

    def _is_rate_limit_error(self, error: Exception) -> bool:
        if getattr(error, "status_code", None) == 429:
            return True
        if error.__class__.__name__ == "RateLimitError":
            return True
        message = str(error).lower()
        return "rate limit" in message or "too many requests" in message

    def _rate_limit_sleep_seconds(self, error: Exception) -> float:
        match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", str(error), re.IGNORECASE)
        if match is not None:
            return float(match.group(1)) + 0.5
        return self.runtime_config.default_rate_limit_backoff_seconds

    async def _wait_for_model_slot(self) -> None:
        inter_call_delay = self.runtime_config.inter_call_delay_seconds
        if inter_call_delay <= 0:
            return

        loop = asyncio.get_running_loop()
        async with self._invoke_spacing_lock:
            now = loop.time()
            if now < self._next_model_call_time:
                await asyncio.sleep(self._next_model_call_time - now)
                now = loop.time()
            self._next_model_call_time = now + inter_call_delay

    async def _invoke_structured(self, role_name: str, schema, messages):
        base_max_tokens = self.runtime_config.role_config(role_name).max_tokens
        current_max_tokens = base_max_tokens
        rate_limit_attempt = 0
        length_limit_attempt = 0

        while True:
            model = self._build_role_model_for_attempt(
                role_name,
                None if current_max_tokens == base_max_tokens else current_max_tokens,
            )
            if model is None:
                raise RuntimeError("{} model is unavailable".format(role_name))
            runnable = model.with_structured_output(schema)

            try:
                async with self._invoke_semaphore:
                    await self._wait_for_model_slot()
                    return await runnable.ainvoke(messages)
            except Exception as error:
                if self._is_rate_limit_error(error) and rate_limit_attempt < self.runtime_config.max_rate_limit_retries:
                    sleep_seconds = self._rate_limit_sleep_seconds(error)
                    self.logger.info(
                        "Rate limited during %s invocation; sleeping %.2fs before retry %s/%s",
                        role_name,
                        sleep_seconds,
                        rate_limit_attempt + 1,
                        self.runtime_config.max_rate_limit_retries,
                    )
                    rate_limit_attempt += 1
                    await asyncio.sleep(sleep_seconds)
                    continue

                if self._is_length_limit_error(error) and length_limit_attempt < self.runtime_config.max_length_limit_retries:
                    next_max_tokens = self._next_length_retry_tokens(current_max_tokens)
                    if next_max_tokens is not None:
                        self.logger.info(
                            "Length-limited during %s invocation; increasing max_tokens from %s to %s and retrying %s/%s",
                            role_name,
                            current_max_tokens,
                            next_max_tokens,
                            length_limit_attempt + 1,
                            self.runtime_config.max_length_limit_retries,
                        )
                        current_max_tokens = next_max_tokens
                        length_limit_attempt += 1
                        rate_limit_attempt = 0
                        continue

                raise

    @staticmethod
    def _header_hints_to_dicts(header_hints):
        return [{header_hint.name: header_hint.value} for header_hint in header_hints]

    def _materialize_structured_draft(
        self,
        structured_draft: StructuredArtifactDraft,
        plan_revision: int,
    ) -> ArtifactDraft:
        try:
            content_model = json.loads(structured_draft.content_model_json)
        except json.JSONDecodeError as error:
            raise ValueError("content_model_json must be valid JSON") from error
        if not isinstance(content_model, dict):
            raise ValueError("content_model_json must decode to an object")

        return ArtifactDraft(
            artifact_id=structured_draft.artifact_id,
            path=structured_draft.path,
            kind=structured_draft.kind,
            content_model=content_model,
            headers_hint=self._header_hints_to_dicts(structured_draft.headers_hint),
            review_notes=structured_draft.review_notes,
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
            "stylesheet": {
                "rules": [{"selector": "<css selector>", "declarations": {"<property>": "<value>"}}],
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

    def _build_review_evidence(self, bundle: GeneratedBundle) -> str:
        total_budget = max(512, self.runtime_config.max_tool_response_chars)
        per_artifact_budget = max(180, total_budget // max(1, len(bundle.artifacts)))
        evidence_sections = []
        for artifact in bundle.artifacts:
            body_preview = artifact.body_bytes.decode("utf-8", errors="replace").strip()
            if len(body_preview) > per_artifact_budget:
                body_preview = body_preview[:per_artifact_budget] + "...[truncated]"
            evidence_sections.append(
                "Path: {path}\nKind: {kind}\nHeaders: {headers}\nPreview:\n{preview}".format(
                    path=artifact.path,
                    kind=artifact.kind,
                    headers=artifact.headers,
                    preview=body_preview or "<empty>",
                )
            )
        return "\n\n".join(evidence_sections)

    async def _normalize_request_node(self, state: GraphState):
        request = GenerationRequest.model_validate(state["request"])
        return {
            "request": request,
            "review_iteration": 0,
            "review_feedback": [],
            "plan_revision": 0,
            "trace_notes": ["normalized {}".format(request.normalized_path)],
            "errors": [],
        }

    async def _expert_node(self, state: GraphState):
        request = state["request"]
        heuristic_spec = await self._heuristic_expert_spec(request)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Expert role in a honeypot resource generator. "
                    "Infer attacker intent from the requested path and return a concise structured spec."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Request path: {path}\n"
                    "Host: {host}\n"
                    "Index page: {index_page}\n"
                    "Heuristic intent seed: {intent}\n"
                    "Return an ExpertSpec that improves or confirms the seed."
                ).format(
                    path=request.normalized_path,
                    host=request.host or "unknown",
                    index_page=request.index_page,
                    intent=heuristic_spec.intent_family,
                ),
            },
        ]

        try:
            expert_spec = await self._invoke_structured("expert", ExpertSpec, messages)
            expert_spec = ExpertSpec.model_validate(expert_spec)
        except Exception as error:
            self.logger.info("Falling back to heuristic expert spec for %s: %s", request.normalized_path, error)
            expert_spec = heuristic_spec

        return {"expert_spec": expert_spec, "trace_notes": ["expert:{}".format(expert_spec.intent_family)]}

    async def _design_node(self, state: GraphState):
        request = state["request"]
        expert_spec = state["expert_spec"]
        review_feedback = state.get("review_feedback", [])
        plan_revision = state.get("plan_revision", 0) + 1
        heuristic_plan = self._heuristic_plan(request, expert_spec, review_feedback)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Design role in a honeypot bundle generator. "
                    "Plan a small static-only bundle that satisfies the requested path and its nearby context."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Request path: {path}\n"
                    "Intent family: {intent}\n"
                    "Attacker goal: {goal}\n"
                    "Review feedback: {feedback}\n"
                    "Bundle size cap: {count} artifacts, {bytes} bytes\n"
                    "Additional planning rule: {rule}\n"
                    "Return a ResourcePlan that keeps every artifact static_file only."
                ).format(
                    path=request.normalized_path,
                    intent=expert_spec.intent_family,
                    goal=expert_spec.attacker_goal,
                    feedback=", ".join(review_feedback) if review_feedback else "none",
                    count=self.runtime_config.max_bundle_artifacts,
                    bytes=self.runtime_config.max_bundle_bytes,
                    rule=self._design_guardrails_for_intent(expert_spec.intent_family),
                ),
            },
        ]

        try:
            resource_plan = await self._invoke_structured("design", ResourcePlan, messages)
            resource_plan = ResourcePlan.model_validate(resource_plan)
            validate_plan(resource_plan, request, self.runtime_config)
        except Exception as error:
            self.logger.info("Falling back to heuristic resource plan for %s: %s", request.normalized_path, error)
            resource_plan = heuristic_plan

        return {
            "resource_plan": resource_plan,
            "review_feedback": [],
            "plan_revision": plan_revision,
            "trace_notes": ["design:{}:{}".format(resource_plan.primary_path, len(resource_plan.artifacts))],
        }

    def _fan_out_coders(self, state: GraphState):
        request = state["request"]
        expert_spec = state["expert_spec"]
        resource_plan = state["resource_plan"]
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
                        "pending_artifact": artifact,
                        "plan_revision": plan_revision,
                    },
                )
            )
        return sends

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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Coder role generating one static artifact draft. "
                    "Return a StructuredArtifactDraft only. "
                    "The content_model_json field must be a valid JSON object encoded as a string. "
                    "headers_hint must be a list of {name,value} objects. Do not invent additional files. "
                    "Never use internal planning words such as fake, lure, attacker, attackers, or honeypot in served text, comments, titles, footers, or file content."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Primary request: {path}\n"
                    "Artifact path: {artifact_path}\n"
                    "Artifact kind: {kind}\n"
                    "Artifact purpose: {purpose}\n"
                    "Related links: {links}\n"
                    "Environment theme: {theme}\n"
                    "Use this abstract content_model skeleton only as a shape guide; replace every placeholder with concrete artifact content: {content_model_skeleton}"
                ).format(
                    path=request.normalized_path,
                    artifact_path=artifact.path,
                    kind=artifact.kind,
                    purpose=artifact.purpose,
                    links=", ".join(artifact.links_to) if artifact.links_to else "none",
                    theme=expert_spec.environment_theme,
                    content_model_skeleton=content_model_skeleton,
                ),
            },
        ]

        try:
            structured_draft = await self._invoke_structured("coder", StructuredArtifactDraft, messages)
            structured_draft = StructuredArtifactDraft.model_validate(structured_draft)
            draft = self._materialize_structured_draft(structured_draft, plan_revision)
            validate_artifact_draft(draft, request)
        except Exception as error:
            self.logger.info("Falling back to heuristic draft for %s: %s", artifact.path, error)
            draft = heuristic_draft

        return {
            "artifact_drafts": [draft],
            "trace_notes": ["coder:{}".format(draft.path)],
        }

    async def _assemble_bundle(self, state: GraphState):
        request = state["request"]
        resource_plan = state["resource_plan"]
        current_revision = state.get("plan_revision", 0)
        current_drafts = [
            draft for draft in state.get("artifact_drafts", []) if draft.plan_revision == current_revision
        ]
        if not current_drafts:
            raise ValidationError("no artifact drafts were produced for plan revision {}".format(current_revision))

        artifacts = []
        for draft in current_drafts:
            validate_artifact_draft(draft, request)
            artifacts.append(render_artifact(draft))

        bundle = GeneratedBundle(
            primary_path=resource_plan.primary_path,
            artifacts=artifacts,
            review_summary="awaiting review",
            used_fallback=False,
        )
        return {
            "generated_bundle": bundle,
            "trace_notes": ["assembled:{}".format(len(bundle.artifacts))],
        }

    async def _review_node(self, state: GraphState):
        request = state["request"]
        expert_spec = state["expert_spec"]
        resource_plan = state["resource_plan"]
        bundle = state["generated_bundle"]
        issues = list(state.get("errors", []))

        try:
            validate_plan(resource_plan, request, self.runtime_config)
            validate_bundle(bundle, request, self.runtime_config)
        except ValidationError as error:
            issues.append(str(error))

        if issues:
            return self._review_revise_or_fallback(state, issues)

        review_evidence = self._build_review_evidence(bundle)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Review role in a honeypot generator. "
                    "Approve only if the bundle is coherent, internally linked, and convincing for the inferred probe. "
                    "Use the provided bundle evidence, not just the paths."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Intent family: {intent}\n"
                    "Primary path: {primary}\n"
                    "Bundle paths: {paths}\n"
                    "Review focus: {focus}\n\n"
                    "Bundle evidence:\n{evidence}\n\n"
                    "Return ReviewDecision with approve, revise, or fallback."
                ).format(
                    intent=expert_spec.intent_family,
                    primary=bundle.primary_path,
                    paths=", ".join(artifact.path for artifact in bundle.artifacts),
                    focus=", ".join(resource_plan.review_focus) if resource_plan.review_focus else "coherence",
                    evidence=review_evidence,
                ),
            },
        ]

        try:
            decision = await self._invoke_structured("review", ReviewDecision, messages)
            decision = ReviewDecision.model_validate(decision)
        except Exception as error:
            self.logger.info("Falling back to deterministic review for %s: %s", request.normalized_path, error)
            decision = ReviewDecision(decision="approve", reasons=["deterministic validation passed"], required_fixes=[])

        if decision.decision == "approve":
            return {"review_decision": decision, "trace_notes": ["review:approve"]}

        return self._review_revise_or_fallback(state, decision.required_fixes or decision.reasons)

    def _review_revise_or_fallback(self, state: GraphState, reasons: list[str]):
        next_iteration = state.get("review_iteration", 0) + 1
        if next_iteration >= self.runtime_config.max_review_loops:
            decision = ReviewDecision(decision="fallback", reasons=reasons, required_fixes=reasons)
            route = "fallback"
        else:
            decision = ReviewDecision(decision="revise", reasons=reasons, required_fixes=reasons)
            route = "revise"
        return {
            "review_decision": decision,
            "review_iteration": next_iteration,
            "review_feedback": reasons,
            "trace_notes": ["review:{}".format(route)],
        }

    def _route_after_review(self, state: GraphState):
        decision = state.get("review_decision")
        if decision is None:
            return "fallback"
        return decision.decision

    async def _finalize_bundle(self, state: GraphState):
        request = state["request"]
        bundle = GeneratedBundle.model_validate(state["generated_bundle"])
        review_decision = state.get("review_decision")
        artifacts = sorted(
            bundle.artifacts,
            key=lambda artifact: (artifact.path != bundle.primary_path, artifact.path),
        )
        finalized = bundle.model_copy(
            update={
                "artifacts": artifacts,
                "review_summary": "; ".join(review_decision.reasons) if review_decision and review_decision.reasons else bundle.review_summary,
            }
        )
        validate_bundle(finalized, request, self.runtime_config)
        return {"generated_bundle": finalized, "trace_notes": ["finalized"]}

    async def _fallback_node(self, state: GraphState):
        request = state["request"]
        expert_spec = state.get("expert_spec")
        review_decision = state.get("review_decision")
        reasons = review_decision.reasons if review_decision else ["graph fell back without review decision"]
        bundle = build_fallback_bundle(
            request,
            expert_spec=expert_spec,
            reasons=reasons,
            max_artifacts=self.runtime_config.max_bundle_artifacts,
        )
        validate_bundle(bundle, request, self.runtime_config)
        return {"generated_bundle": bundle, "trace_notes": ["fallback"]}

    async def _heuristic_expert_spec(self, request: GenerationRequest) -> ExpertSpec:
        intent_family = infer_intent_family(request.normalized_path)
        research = None
        if self.runtime_config.enable_live_research:
            research_query = self._research_query_for_intent(intent_family, request.normalized_path)
            research = await asyncio.to_thread(web_research, research_query, self.runtime_config)

        theme_by_intent = {
            "config_theft": "Production application secrets",
            "cms_probe": "WordPress administrative portal",
            "backup_probe": "Operational backup inventory",
            "admin_portal": "Internal administrative dashboard",
            "framework_probe": "Application framework reconnaissance",
            "generic_recon": "Generic service portal",
        }
        goal_by_intent = {
            "config_theft": "Obtain credentials, hostnames, and secret material from leaked configuration files.",
            "cms_probe": "Locate a believable CMS login surface and linked assets for credential capture.",
            "backup_probe": "Discover archived copies, export listings, or backup manifests with operational clues.",
            "admin_portal": "Reach an administrative login or dashboard surface with nearby supporting assets.",
            "framework_probe": "Verify the underlying stack and enumerate framework-specific resources.",
            "generic_recon": "Map the service and test whether nearby resources disclose useful context.",
        }
        kind_by_intent = {
            "config_theft": "config_text",
            "cms_probe": "html_page",
            "backup_probe": "backup_manifest",
            "admin_portal": "html_page",
            "framework_probe": "html_page",
            "generic_recon": "html_page",
        }
        references = research.references if research is not None else []
        supporting_context = research.snippets if research is not None and research.snippets else self._supporting_context(intent_family)

        return ExpertSpec(
            intent_family=intent_family,
            attacker_goal=goal_by_intent[intent_family],
            confidence=0.88,
            primary_resource_kind=kind_by_intent[intent_family],
            lure_requirements=self._lure_requirements(intent_family),
            supporting_context=supporting_context,
            environment_theme=theme_by_intent[intent_family],
            references=references,
        )

    def _heuristic_plan(
        self,
        request: GenerationRequest,
        expert_spec: ExpertSpec,
        review_feedback: list[str] | None = None,
    ) -> ResourcePlan:
        review_feedback = review_feedback or []
        primary_path = request.normalized_path
        intent_family = expert_spec.intent_family

        if intent_family == "config_theft":
            backup_path = primary_path + ".bak" if not primary_path.endswith(".bak") else primary_path + ".old"
            artifacts = [
                PlannedArtifact(
                    artifact_id="primary-config",
                    path=primary_path,
                    kind="config_text",
                    purpose="Primary application environment file",
                ),
                PlannedArtifact(
                    artifact_id="config-log",
                    path="/storage/logs/app.log",
                    kind="log_excerpt",
                    purpose="Operational log excerpt referencing the application environment",
                    depends_on=["primary-config"],
                ),
                PlannedArtifact(
                    artifact_id="config-backup",
                    path=backup_path,
                    kind="backup_manifest",
                    purpose="Backup manifest adjacent to the application environment file",
                    depends_on=["primary-config"],
                ),
            ]
        elif intent_family == "cms_probe":
            stylesheet_path = "/wp-content/themes/twentytwenty/style.css"
            script_path = "/wp-includes/js/wp-login.js"
            helper_path = "/wp-login.php"
            artifacts = [
                PlannedArtifact(
                    artifact_id="wp-login",
                    path=primary_path,
                    kind="html_page",
                    purpose="Primary WordPress login page",
                    links_to=[stylesheet_path, script_path, helper_path],
                ),
                PlannedArtifact(
                    artifact_id="wp-style",
                    path=stylesheet_path,
                    kind="stylesheet",
                    purpose="Stylesheet for the WordPress login page",
                ),
                PlannedArtifact(
                    artifact_id="wp-script",
                    path=script_path,
                    kind="javascript",
                    purpose="Minimal login-page behavior",
                ),
                PlannedArtifact(
                    artifact_id="wp-helper",
                    path=helper_path,
                    kind="html_page",
                    purpose="Password reset helper page",
                    links_to=[stylesheet_path, primary_path],
                ),
            ]
        elif intent_family == "backup_probe":
            artifacts = [
                PlannedArtifact(
                    artifact_id="backup-primary",
                    path=primary_path,
                    kind="backup_manifest",
                    purpose="Primary backup manifest for the requested archive",
                ),
                PlannedArtifact(
                    artifact_id="backup-index",
                    path="/index.html",
                    kind="html_page",
                    purpose="Landing page that links to operational resources",
                    links_to=[primary_path, "/robots.txt"],
                ),
                PlannedArtifact(
                    artifact_id="backup-robots",
                    path="/robots.txt",
                    kind="robots_txt",
                    purpose="Robots policy with restricted backup areas",
                ),
            ]
        else:
            artifacts = [
                PlannedArtifact(
                    artifact_id="primary-page",
                    path=primary_path,
                    kind="html_page",
                    purpose="Primary service portal page",
                    links_to=["/robots.txt"],
                ),
                PlannedArtifact(
                    artifact_id="robots",
                    path="/robots.txt",
                    kind="robots_txt",
                    purpose="Robots policy for auxiliary context",
                ),
                PlannedArtifact(
                    artifact_id="sitemap",
                    path="/sitemap.xml",
                    kind="sitemap_xml",
                    purpose="Sitemap describing the nearby static resources",
                ),
            ]

        if any("robots" in feedback for feedback in review_feedback) and not any(a.path == "/robots.txt" for a in artifacts):
            artifacts.append(
                PlannedArtifact(
                    artifact_id="robots-extra",
                    path="/robots.txt",
                    kind="robots_txt",
                    purpose="Added to satisfy review feedback on linked support files",
                )
            )

        return ResourcePlan(
            primary_path=primary_path,
            theme_summary=expert_spec.environment_theme,
            artifacts=artifacts[: self.runtime_config.max_bundle_artifacts],
            bundle_budget_count=min(len(artifacts), self.runtime_config.max_bundle_artifacts),
            bundle_budget_bytes=self.runtime_config.max_bundle_bytes,
            static_only=True,
            review_focus=["static-only", "internal-link-completeness", "theme-consistency"],
        )

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
                "paragraphs": list(expert_spec.lure_requirements[:2]) + list(expert_spec.supporting_context[:1]),
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

        draft = ArtifactDraft(
            artifact_id=artifact.artifact_id,
            path=artifact.path,
            kind=artifact.kind,
            content_model=content_model,
            headers_hint=[{"X-Tanner-Generated": "agentic"}],
            review_notes=[artifact.purpose],
            plan_revision=plan_revision,
        )
        validate_artifact_draft(draft, request)
        return draft

    @staticmethod
    def _research_query_for_intent(intent_family: str, normalized_path: str) -> str:
        if intent_family == "cms_probe":
            return "wordpress login screen {}".format(normalized_path)
        if intent_family == "config_theft":
            return "dotenv configuration example {}".format(normalized_path)
        if intent_family == "backup_probe":
            return "backup manifest export listing {}".format(normalized_path)
        return "service portal {}".format(normalized_path)

    @staticmethod
    def _supporting_context(intent_family: str) -> list[str]:
        defaults = {
            "config_theft": [
                "The lure should expose realistic database, cache, and mail settings.",
                "Adjacent files should imply a live production environment rather than a toy sample.",
            ],
            "cms_probe": [
                "The primary page should feel like a WordPress administrative surface with familiar assets.",
                "Linked support files should resolve locally and avoid dynamic server behavior.",
            ],
            "backup_probe": [
                "The bundle should hint at operational exports and recovery workflows.",
                "Supporting files should reinforce that backups are archived on the same host.",
            ],
            "admin_portal": [
                "The portal should look restricted and business-facing.",
                "Supporting files should reinforce login and navigation realism.",
            ],
            "framework_probe": [
                "The bundle should reveal framework-adjacent structure without needing code execution.",
                "Supporting files should stay static and coherent.",
            ],
            "generic_recon": [
                "The bundle should give the attacker enough context to keep exploring.",
                "Every referenced internal path must exist in the generated bundle.",
            ],
        }
        return defaults[intent_family]

    @staticmethod
    def _lure_requirements(intent_family: str) -> list[str]:
        requirements = {
            "config_theft": [
                "Expose realistic secret-looking values and infrastructure hostnames.",
                "Keep the file compact enough to look harvested from a real deployment.",
            ],
            "cms_probe": [
                "Match common WordPress login terminology and page structure.",
                "Reference nearby static assets that SNARE can persist and serve later.",
            ],
            "backup_probe": [
                "Imply a recent backup workflow with concrete file names.",
                "Keep the artifact static and parseable as a leaked archive index.",
            ],
            "admin_portal": [
                "Use an internal-system tone that suggests authenticated access.",
                "Render as a static login or dashboard page with believable labels.",
            ],
            "framework_probe": [
                "Expose enough structure to suggest a framework-specific surface.",
                "Avoid dynamic behavior or server-side execution assumptions.",
            ],
            "generic_recon": [
                "Provide a believable static entry point for further browsing.",
                "Add at least one adjacent support artifact so the bundle feels contextual.",
            ],
        }
        return requirements[intent_family]

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
        return AgenticBundleGenerator._page_heading_for_artifact(normalized)

    @staticmethod
    def _design_guardrails_for_intent(intent_family: str) -> str:
        if intent_family == "config_theft":
            return (
                "Include the primary leaked configuration file plus at least one adjacent supporting decoy such as a log excerpt, "
                "backup manifest, or alternate config artifact. Never use internal planning words in served content."
            )
        return "Keep the bundle compact, coherent, and free of internal planning language."

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

    async def generate_bundle(self, host, path, site_profile):
        request = ensure_generation_request(host, path, site_profile if isinstance(site_profile, dict) else {})
        initial_state: GraphState = {
            "request": request,
            "artifact_drafts": [],
            "review_feedback": [],
            "review_iteration": 0,
            "trace_notes": [],
            "errors": [],
            "plan_revision": 0,
        }
        thread_id = "meta:{}:{}".format(request.normalized_path, uuid.uuid4())

        try:
            async with AsyncSqliteSaver.from_conn_string(self.runtime_config.checkpoint_path) as checkpointer:
                if not hasattr(checkpointer.conn, "is_alive"):
                    checkpointer.conn.is_alive = lambda: True
                graph = self._graph_builder.compile(checkpointer=checkpointer)
                result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
            bundle = GeneratedBundle.model_validate(result["generated_bundle"])
            validate_bundle(bundle, request, self.runtime_config)
            return bundle
        except Exception as error:
            self.logger.exception("Agentic bundle generation failed for %s", request.normalized_path)
            return build_fallback_bundle(
                request,
                reasons=[str(error)],
                max_artifacts=self.runtime_config.max_bundle_artifacts,
            )
