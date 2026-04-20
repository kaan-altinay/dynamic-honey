from __future__ import annotations

import base64
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
    ArtifactReferenceContext,
    ExpertSpec,
    FetchedAsset,
    GeneratedArtifact,
    GeneratedBundle,
    GenerationRequest,
    GeneratorRuntimeConfig,
    PlannedArtifact,
    ReferencePage,
    ReferencePack,
    ResourcePlan,
    ReviewDecision,
    StructuredBackupManifestDraft,
    StructuredBinaryAssetDraft,
    StructuredConfigTextDraft,
    StructuredCredentialBaitDraft,
    StructuredHtmlPageDraft,
    StructuredJavascriptDraft,
    StructuredJsonDocumentDraft,
    StructuredLogExcerptDraft,
    StructuredPlainTextDraft,
    StructuredReviewDecision,
    StructuredRobotsTxtDraft,
    StructuredSitemapDraft,
    StructuredStylesheetDraft,
    StructuredXmlDocumentDraft,
)
from tanner.generator.agentic.renderers import render_artifact
from tanner.generator.agentic.tools import fetch_reference_page, fetch_static_asset, web_research
from tanner.generator.agentic.validators import (
    ValidationError,
    _is_external_reference,
    _normalize_allowed_paths,
    ensure_generation_request,
    infer_intent_family,
    normalize_path,
    validate_artifact_draft,
    validate_artifact_draft_contract,
    validate_bundle,
    validate_plan,
 )
from tanner.generator.base_generator import BaseGenerator


class GraphState(TypedDict, total=False):
    request: GenerationRequest
    expert_spec: ExpertSpec
    resource_plan: ResourcePlan
    reference_pack: ReferencePack
    pending_artifact: PlannedArtifact
    artifact_drafts: Annotated[list[ArtifactDraft], operator.add]
    review_decision: ReviewDecision
    review_iteration: int
    design_validation_iteration: int
    design_validation_decision: str
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
        builder.add_node("design_gate_node", self._design_gate_node)
        builder.add_node("prepare_reference_pack", self._prepare_reference_pack)
        builder.add_node("coder_node", self._coder_node)
        builder.add_node("assemble_bundle", self._assemble_bundle)
        builder.add_node("review_node", self._review_node)
        builder.add_node("finalize_bundle", self._finalize_bundle)
        builder.add_node("fallback_node", self._fallback_node)

        builder.add_edge(START, "normalize_request")
        builder.add_edge("normalize_request", "expert_node")
        builder.add_edge("expert_node", "design_node")
        builder.add_edge("design_node", "design_gate_node")
        builder.add_conditional_edges(
            "design_gate_node",
            self._route_after_design_gate,
            {"approve": "prepare_reference_pack", "revise": "design_node", "fallback": "fallback_node"},
        )
        builder.add_conditional_edges("prepare_reference_pack", self._fan_out_coders, ["coder_node"])
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
        return "length limit was reached" in message or "max completion tokens reached" in message

    @staticmethod
    def _is_json_validate_failed_error(error: Exception) -> bool:
        error_code = getattr(error, "code", None)
        if error_code == "json_validate_failed":
            return True

        body = getattr(error, "body", None)
        if isinstance(body, dict):
            error_payload = body.get("error")
            if isinstance(error_payload, dict) and error_payload.get("code") == "json_validate_failed":
                return True

        message = str(error).lower()
        return "json_validate_failed" in message or "failed to validate json" in message

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

                if self._is_json_validate_failed_error(error) and length_limit_attempt < self.runtime_config.max_length_limit_retries:
                    self.logger.info(
                        "Structured JSON validation failed during %s invocation; retrying original payload %s/%s",
                        role_name,
                        length_limit_attempt + 1,
                        self.runtime_config.max_length_limit_retries,
                    )
                    length_limit_attempt += 1
                    rate_limit_attempt = 0
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

        return ArtifactDraft(
            artifact_id=artifact.artifact_id,
            path=artifact.path,
            kind=artifact.kind,
            content_model=self._normalize_structured_content_model(
                artifact.kind,
                structured_draft.content_model,
            ),
            headers_hint=self._header_hints_to_dicts(structured_draft.headers_hint),
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

    def _build_review_evidence(self, bundle: GeneratedBundle) -> str:
        """Extract structured features from artifacts for quality review."""
        evidence_sections = []
        per_artifact_budget = max(1024, 2048 // max(1, len(bundle.artifacts)))
        
        for artifact in bundle.artifacts:
            body_text = artifact.body_bytes.decode("utf-8", errors="replace").strip()
            
            # Extract structured features based on kind
            features = []
            if artifact.kind == "html_page":
                # Extract form field names, nav links, CSS classes
                import re
                form_fields = re.findall(r'name=["\']([^"\'>]+)["\']', body_text)
                nav_hrefs = re.findall(r'<a[^>]+href=["\']([^"\'>]+)["\']', body_text)
                css_classes = re.findall(r'class=["\']([^"\'>]+)["\']', body_text)
                title_match = re.search(r'<title>([^<]+)</title>', body_text)
                heading_match = re.search(r'<h[1-6][^>]*>([^<]+)</h[1-6]>', body_text)
                
                features.append(f"Title: {title_match.group(1) if title_match else 'none'}")
                features.append(f"Heading: {heading_match.group(1) if heading_match else 'none'}")
                features.append(f"Form fields: {', '.join(form_fields[:5]) if form_fields else 'none'}")
                features.append(f"Nav links: {', '.join(nav_hrefs[:5]) if nav_hrefs else 'none'}")
                features.append(f"CSS classes: {', '.join(set(css_classes[:8])) if css_classes else 'none'}")
            elif artifact.kind == "config_text":
                # Extract config keys
                lines = body_text.split('\n')[:15]
                keys = []
                for line in lines:
                    if '=' in line and not line.strip().startswith('#'):
                        key = line.split('=')[0].strip()
                        if key:
                            keys.append(key)
                features.append(f"Config keys: {', '.join(keys[:8]) if keys else 'none'}")
                features.append(f"Total entries: {len(keys)}")
            elif artifact.kind == "stylesheet":
                # Extract CSS selectors
                import re
                selectors = re.findall(r'([.#\w-]+)\s*{', body_text)
                features.append(f"CSS selectors: {', '.join(set(selectors[:10])) if selectors else 'none'}")
            
            # Include body preview
            body_preview = body_text[:per_artifact_budget]
            if len(body_text) > per_artifact_budget:
                body_preview += "...[truncated]"
            
            evidence_sections.append(
                "Path: {path}\nKind: {kind}\nHeaders: {headers}\n{features}\n\nContent preview:\n{preview}".format(
                    path=artifact.path,
                    kind=artifact.kind,
                    headers=artifact.headers,
                    features='\n'.join(features) if features else 'No structured features extracted',
                    preview=body_preview or "<empty>",
                )
            )
        return "\n\n---\n\n".join(evidence_sections)

    @staticmethod
    def _normalize_review_decision(structured_decision: StructuredReviewDecision) -> ReviewDecision:
        return ReviewDecision(
            decision=structured_decision.decision,
            reasons=structured_decision.reasons or [],
            required_fixes=structured_decision.required_fixes or [],
        )

    @staticmethod
    def _reference_query_for_intent(expert_spec: ExpertSpec, request: GenerationRequest) -> str:
        intent_queries = {
            "cms_probe": "wordpress login page {}".format(request.normalized_path),
            "config_theft": "dotenv configuration example {}".format(request.normalized_path),
            "backup_probe": "backup manifest example {}".format(request.normalized_path),
            "admin_portal": "admin login page {}".format(request.normalized_path),
            "framework_probe": "framework login page {}".format(request.normalized_path),
            "generic_recon": "service portal page {}".format(request.normalized_path),
        }
        return intent_queries.get(expert_spec.intent_family, request.normalized_path)

    async def _design_reference_candidates(
        self,
        request: GenerationRequest,
        expert_spec: ExpertSpec,
    ) -> list[ReferencePage]:
        candidate_urls = []
        for url in expert_spec.references:
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                candidate_urls.append(url)
        if not candidate_urls and self.runtime_config.enable_live_research:
            research = await asyncio.to_thread(
                web_research,
                self._reference_query_for_intent(expert_spec, request),
                self.runtime_config,
            )
            candidate_urls.extend(research.references)

        reference_pages = []
        seen = set()
        for url in candidate_urls[:3]:
            if url in seen:
                continue
            seen.add(url)
            reference_pages.append(await asyncio.to_thread(fetch_reference_page, url, self.runtime_config))
        return reference_pages

    @staticmethod
    def _summarize_reference_pages(reference_pages: list[ReferencePage]) -> str:
        if not reference_pages:
            return "none"
        summaries = []
        for reference_page in reference_pages:
            candidate_bits = [
                "{}:{}:{}".format(candidate.kind, candidate.local_path_hint, candidate.note or candidate.tag)
                for candidate in reference_page.asset_candidates[:6]
            ]
            summaries.append(
                "URL: {url}\nTitle: {title}\nExcerpt: {excerpt}\nAsset candidates: {candidates}".format(
                    url=reference_page.final_url or reference_page.url,
                    title=reference_page.title or "<untitled>",
                    excerpt=reference_page.text_excerpt[:600] or "<empty>",
                    candidates=", ".join(candidate_bits) if candidate_bits else "none",
                )
            )
        return "\n\n".join(summaries)

    @staticmethod
    def _summarize_reference_pages_compact(reference_pages: list[ReferencePage]) -> str:
        if not reference_pages:
            return "none"
        summaries = []
        for reference_page in reference_pages:
            summaries.append(
                "URL: {url}\nTitle: {title}".format(
                    url=reference_page.final_url or reference_page.url,
                    title=reference_page.title or "<untitled>",
                )
            )
        return "\n\n".join(summaries)

    @staticmethod
    def _reference_context_for_artifact(reference_pack: ReferencePack, artifact_id: str) -> ArtifactReferenceContext | None:
        for artifact_context in reference_pack.artifact_contexts:
            if artifact_context.artifact_id == artifact_id:
                return artifact_context
        return None

    @staticmethod
    def _generated_artifacts_from_reference_pack(reference_pack: ReferencePack) -> list[GeneratedArtifact]:
        generated_assets = []
        for fetched_asset in reference_pack.fetched_assets:
            generated_assets.append(
                GeneratedArtifact(
                    path=fetched_asset.local_path,
                    kind="asset_file",
                    headers=[{"Content-Type": fetched_asset.content_type}],
                    body_bytes=fetched_asset.body_bytes,
                    status_code=200,
                    source_artifact_id=fetched_asset.asset_id,
                    artifact_scope="static_file",
                )
            )
        return generated_assets

    async def _normalize_request_node(self, state: GraphState):
        request = GenerationRequest.model_validate(state["request"])
        return {
            "request": request,
            "review_iteration": 0,
            "design_validation_iteration": 0,
            "design_validation_decision": "approve",
            "review_feedback": [],
            "plan_revision": 0,
            "trace_notes": ["normalized {}".format(request.normalized_path)],
            "errors": [],
        }

    async def _expert_node(self, state: GraphState):
        request = state["request"]
        heuristic_spec = await self._heuristic_expert_spec(request)
        research_snippets = "none"
        research_links = "none"
        if self.runtime_config.enable_live_research:
            research = await asyncio.to_thread(
                web_research,
                self._reference_query_for_intent(heuristic_spec, request),
                self.runtime_config,
            )
            research_snippets = " | ".join(research.snippets) if research.snippets else "none"
            research_links = ", ".join(research.references) if research.references else "none"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Expert role in a honeypot resource generator. "
                    "Infer attacker intent from the requested path and return a concise structured spec.\n\n"
                    "Intent families (choose one):\n"
                    "- config_theft: Attacker seeks leaked configuration files with credentials/secrets\n"
                    "- cms_probe: Attacker probes for CMS admin login surfaces (WordPress, Drupal, etc.)\n"
                    "- admin_portal: Attacker seeks generic administrative dashboards or control panels\n"
                    "- framework_probe: Attacker enumerates framework-specific resources (Laravel, Django, etc.)\n"
                    "- backup_probe: Attacker searches for backup archives, manifests, or export listings\n"
                    "- generic_recon: General reconnaissance of service structure and nearby resources"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Request path: {path}\n"
                    "Host: {host}\n"
                    "Index page: {index_page}\n"
                    "Heuristic intent seed: {intent}\n"
                    "Search snippets: {snippets}\n"
                    "Candidate reference URLs: {links}\n"
                    "Return an ExpertSpec that improves or confirms the seed.\n\n"
                    "Field guidance:\n"
                    "- references: must be URLs of live pages whose structure can be fetched and studied, not documentation links.\n"
                    "- lure_requirements and supporting_context: internal directives for downstream generators; they will never be served to users directly.\n"
                    "- environment_theme: should read as a realistic organization or service label, e.g., 'Contoso Employee Portal'."
                ).format(
                    path=request.normalized_path,
                    host=request.host or "unknown",
                    index_page=request.index_page,
                    intent=heuristic_spec.intent_family,
                    snippets=research_snippets,
                    links=research_links,
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

        # Fix 3: Skip LLM design if expert confidence is low
        if expert_spec.confidence < 0.5:
            self.logger.info(
                "Skipping LLM design for %s due to low expert confidence (%.2f < 0.5); using heuristic plan",
                request.normalized_path,
                expert_spec.confidence,
            )
            return {
                "resource_plan": heuristic_plan,
                "review_feedback": [],
                "plan_revision": plan_revision,
                "trace_notes": ["design:heuristic:low_confidence:{}:{}".format(heuristic_plan.primary_path, len(heuristic_plan.artifacts))],
            }
        reference_pages = await self._design_reference_candidates(request, expert_spec)
        reference_summary = self._summarize_reference_pages(reference_pages)
        compact_reference_summary = self._summarize_reference_pages_compact(reference_pages)

        def _build_design_messages(reference_candidates: str):
            return [
                {
                    "role": "system",
                "content": (
                    "You are the Design role in a honeypot bundle generator. "
                    "Plan a static-only bundle that satisfies the requested path and its nearby context.\n\n"
                    "Requirements:\n"
                    "- At least 2 generated artifacts required (one primary, plus supporting artifacts)\n"
                    "- Up to {max_artifacts} total outputs (generated artifacts + fetched assets)\n"
                    "- Choose exactly one representation per local path: generated artifact OR fetched asset, never both\n"
                    "- Generated artifacts must use supported kinds only: html_page, config_text, json_document, plain_text, binary_asset, stylesheet, javascript, robots_txt, sitemap_xml, xml_document, credential_bait, log_excerpt, backup_manifest\n"
                    "- Never use artifact.kind='asset_file' in artifacts; use artifact.kind='binary_asset' for generated local bytes and reference_asset_plan.asset_fetches for copied remote assets\n\n"
                    "Field guidance:\n"
                    "- artifact.purpose: This is the primary instruction sent to the Coder. Write it as a concrete content specification (e.g., 'WordPress login page with username and password form, linking to the theme stylesheet')\n"
                    "- artifact.links_to: List every artifact/asset path this artifact should reference in its rendered output (e.g., a page should list its stylesheet, script, and nav link targets)\n"
                    "- reference_asset_plan.asset_fetches: Remote assets you want copied locally for realism. Use this when you need a file derived from a real external source URL\n"
                    "- reference_asset_plan.reference_urls: List the selected reference pages used to justify the design\n\n"
                    "Always use default values for: render_strategy='deterministic', artifact_scope='static_file', must_exist=true, dynamic_candidate=false, service_candidate=false"
                ).format(max_artifacts=self.runtime_config.max_bundle_artifacts),
                },
                {
                    "role": "user",
                "content": (
                    "Request path: {path}\n"
                    "Intent family: {intent}\n"
                    "Attacker goal: {goal}\n"
                    "Review feedback: {feedback}\n"
                    "Bundle budget: up to {count} total outputs, {bytes} bytes\n\n"
                    "Contract rules (must satisfy all five):\n"
                    "1. No duplicate local paths across artifacts and asset_fetches\n"
                    "2. Every depends_on and required_for_artifact_ids value must reference a valid artifact_id in this plan\n"
                    "3. bundle_budget_count must exactly equal (number of artifacts + number of asset_fetches)\n"
                    "4. artifact.kind='asset_file' is forbidden in artifacts; use artifact.kind='binary_asset' for generated bytes or reference_asset_plan.asset_fetches for copied remote assets\n"
                    "5. Extension-kind contract for primary_path: .xml->xml_document (except /sitemap.xml->sitemap_xml), .json->json_document, .txt->plain_text (except /robots.txt->robots_txt), binary extensions (.ico/.jpg/.jpeg/.png/.gif/.webp/.bmp/.svg/.woff/.woff2/.ttf/.otf)->binary_asset\n\n"
                    "Additional planning rule: {rule}\n\n"
                    "Example ResourcePlan for cms_probe intent (for reference only):\n"
                    "{{\n"
                    "  \"primary_path\": \"/wp-admin/login.php\",\n"
                    "  \"theme_summary\": \"WordPress administrative login portal\",\n"
                    "  \"artifacts\": [\n"
                    "    {{\n"
                    "      \"artifact_id\": \"wp-login-page\",\n"
                    "      \"path\": \"/wp-admin/login.php\",\n"
                    "      \"kind\": \"html_page\",\n"
                    "      \"purpose\": \"WordPress login page with username and password form fields, linking to theme stylesheet and login script\",\n"
                    "      \"links_to\": [\"/wp-content/themes/twentytwenty/style.css\", \"/wp-includes/js/wp-login.js\", \"/wp-login.php\"]\n"
                    "    }},\n"
                    "    {{\n"
                    "      \"artifact_id\": \"wp-theme-style\",\n"
                    "      \"path\": \"/wp-content/themes/twentytwenty/style.css\",\n"
                    "      \"kind\": \"stylesheet\",\n"
                    "      \"purpose\": \"Theme stylesheet with WordPress admin color scheme and form styling\"\n"
                    "    }},\n"
                    "    {{\n"
                    "      \"artifact_id\": \"wp-login-script\",\n"
                    "      \"path\": \"/wp-includes/js/wp-login.js\",\n"
                    "      \"kind\": \"javascript\",\n"
                    "      \"purpose\": \"Login page behavior script for form field focus\"\n"
                    "    }},\n"
                    "    {{\n"
                    "      \"artifact_id\": \"wp-password-reset\",\n"
                    "      \"path\": \"/wp-login.php\",\n"
                    "      \"kind\": \"html_page\",\n"
                    "      \"purpose\": \"Password reset helper page linking back to login and sharing the theme stylesheet\",\n"
                    "      \"links_to\": [\"/wp-content/themes/twentytwenty/style.css\", \"/wp-admin/login.php\"]\n"
                    "    }}\n"
                    "  ],\n"
                    "  \"bundle_budget_count\": 4,\n"
                    "  \"bundle_budget_bytes\": 262144,\n"
                    "  \"static_only\": true\n"
                    "}}\n\n"
                    "Reference page candidates:\n{reference_summary}\n\n"
                    "Return a ResourcePlan with at least 2 generated artifacts that satisfies all contract rules."
                    ).format(
                        path=request.normalized_path,
                        intent=expert_spec.intent_family,
                        goal=expert_spec.attacker_goal,
                        feedback=", ".join(review_feedback) if review_feedback else "none",
                        count=self.runtime_config.max_bundle_artifacts,
                        bytes=self.runtime_config.max_bundle_bytes,
                        rule=self._design_guardrails_for_intent(expert_spec.intent_family),
                        reference_summary=reference_candidates,
                    ),
                },
            ]

        attempt_payloads = [
            ("original", reference_summary),
            ("compact", compact_reference_summary),
        ]
        resource_plan = None
        last_error: Exception | None = None
        for attempt_index, (payload_name, candidate_summary) in enumerate(attempt_payloads, start=1):
            try:
                resource_plan = await self._invoke_structured(
                    "design",
                    ResourcePlan,
                    _build_design_messages(candidate_summary),
                )
                resource_plan = ResourcePlan.model_validate(resource_plan)
                resource_plan = self._normalize_resource_plan(resource_plan, request)
                break
            except Exception as error:
                last_error = error
                if attempt_index < len(attempt_payloads):
                    self.logger.info(
                        "Design planning attempt %s/%s failed for %s using %s payload: %s",
                        attempt_index,
                        len(attempt_payloads),
                        request.normalized_path,
                        payload_name,
                        error,
                    )
                    continue

        if resource_plan is None:
            self.logger.info(
                "Falling back to heuristic resource plan for %s after %s design attempts: %s",
                request.normalized_path,
                len(attempt_payloads),
                last_error,
            )
            resource_plan = heuristic_plan

        return {
            "resource_plan": resource_plan,
            "review_feedback": [],
            "plan_revision": plan_revision,
            "trace_notes": ["design:{}:{}".format(resource_plan.primary_path, len(resource_plan.artifacts))],
        }

    def _design_revise_or_fallback(self, state: GraphState, reasons: list[str]):
        next_iteration = state.get("design_validation_iteration", 0) + 1
        if next_iteration >= self.runtime_config.max_review_loops:
            decision = "fallback"
        else:
            decision = "revise"
        return {
            "design_validation_decision": decision,
            "design_validation_iteration": next_iteration,
            "review_decision": ReviewDecision(decision=decision, reasons=reasons, required_fixes=reasons),
            "review_feedback": reasons,
            "trace_notes": ["design_gate:{}".format(decision)],
        }

    async def _design_gate_node(self, state: GraphState):
        request = state["request"]
        resource_plan = state["resource_plan"]
        try:
            validate_plan(resource_plan, request, self.runtime_config)
        except ValidationError as error:
            self.logger.info(
                "Design plan rejected before coder stage for %s: %s",
                request.normalized_path,
                error,
            )
            return self._design_revise_or_fallback(state, [str(error)])
        return {
            "design_validation_decision": "approve",
            "trace_notes": ["design_gate:approve"],
        }

    def _route_after_design_gate(self, state: GraphState):
        return state.get("design_validation_decision", "fallback")


    @staticmethod
    def _normalize_resource_plan(resource_plan: ResourcePlan, request: GenerationRequest) -> ResourcePlan:
        normalized_artifacts = []
        for artifact in resource_plan.artifacts:
            normalized_links = [
                normalize_path(link, index_page=request.index_page)
                for link in artifact.links_to
                if isinstance(link, str) and link.strip()
            ]
            normalized_artifacts.append(
                artifact.model_copy(
                    update={
                        "path": normalize_path(artifact.path, index_page=request.index_page),
                        "links_to": normalized_links,
                    }
                )
            )

        normalized_asset_fetches = []
        for asset_fetch in resource_plan.reference_asset_plan.asset_fetches:
            normalized_asset_fetches.append(
                asset_fetch.model_copy(
                    update={
                        "local_path": normalize_path(asset_fetch.local_path, index_page=request.index_page),
                    }
                )
            )

        artifact_paths = {artifact.path for artifact in normalized_artifacts}
        unique_asset_fetches = []
        seen_asset_paths = set()
        for asset_fetch in normalized_asset_fetches:
            if asset_fetch.local_path in artifact_paths:
                continue
            if asset_fetch.local_path in seen_asset_paths:
                continue
            seen_asset_paths.add(asset_fetch.local_path)
            unique_asset_fetches.append(asset_fetch)

        normalized_reference_asset_plan = resource_plan.reference_asset_plan.model_copy(
            update={
                "asset_fetches": unique_asset_fetches,
            }
        )

        normalized_output_count = len(normalized_artifacts) + len(unique_asset_fetches)
        normalized_budget_count = normalized_output_count

        return resource_plan.model_copy(
            update={
                "primary_path": normalize_path(resource_plan.primary_path, index_page=request.index_page),
                "artifacts": normalized_artifacts,
                "reference_asset_plan": normalized_reference_asset_plan,
                "bundle_budget_count": normalized_budget_count,
            }
        )

    async def _prepare_reference_pack(self, state: GraphState):
        request = state["request"]
        resource_plan = state["resource_plan"]
        reference_urls = list(resource_plan.reference_asset_plan.reference_urls)
        if not reference_urls:
            reference_urls = list(state["expert_spec"].references[:3])

        reference_pages = []
        seen_urls = set()
        for url in reference_urls[:3]:
            if not isinstance(url, str) or url in seen_urls:
                continue
            seen_urls.add(url)
            reference_pages.append(await asyncio.to_thread(fetch_reference_page, url, self.runtime_config))

        fetched_assets = []
        for asset_fetch in resource_plan.reference_asset_plan.asset_fetches:
            try:
                fetched_asset = await asyncio.to_thread(
                    fetch_static_asset,
                    asset_fetch.source_url,
                    asset_fetch.local_path,
                    asset_fetch.kind,
                    self.runtime_config,
                    asset_fetch.asset_id,
                    asset_fetch.required_for_artifact_ids,
                )
            except Exception as error:
                self.logger.info("Failed to fetch planned asset %s: %s", asset_fetch.source_url, error)
                continue
            fetched_assets.append(fetched_asset)

        artifact_contexts = []
        shared_reference_urls = [reference_page.final_url or reference_page.url for reference_page in reference_pages]
        shared_notes = [reference_page.title for reference_page in reference_pages if reference_page.title]
        allowed_internal_paths = [planned.path for planned in resource_plan.artifacts]

        for artifact in resource_plan.artifacts:
            local_asset_paths = [
                fetched_asset.local_path
                for fetched_asset in fetched_assets
                if artifact.artifact_id in fetched_asset.required_for_artifact_ids
            ]
            artifact_contexts.append(
                ArtifactReferenceContext(
                    artifact_id=artifact.artifact_id,
                    reference_urls=shared_reference_urls,
                    local_asset_paths=local_asset_paths,
                    allowed_local_asset_paths=local_asset_paths,
                    allowed_internal_paths=allowed_internal_paths,
                    primary_path=resource_plan.primary_path,
                    forbidden_external_assets=True,
                    notes=shared_notes,
                )
            )

        reference_pack = ReferencePack(
            reference_pages=reference_pages,
            fetched_assets=fetched_assets,
            artifact_contexts=artifact_contexts,
        )
        return {
            "reference_pack": reference_pack,
            "trace_notes": [
                "references:{}:{}".format(len(reference_pages), len(fetched_assets))
            ],
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

        return ArtifactDraft(
            artifact_id=draft.artifact_id,
            path=draft.path,
            kind=draft.kind,
            content_model=model,
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
                "For json_document: Output a realistic machine-readable JSON object in content_model.document. "
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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Coder role generating one static artifact draft for a honeypot bundle.\n\n"
                    "CRITICAL SAFETY RULE:\n"
                    "Never use internal planning words such as fake, lure, attacker, attackers, honeypot, decoy, or trap in any served text, comments, titles, footers, or file content. The output must read as authentic production content.\n\n"
                    "General instructions:\n"
                    "- Return only the typed structured draft for the requested artifact kind\n"
                    "- artifact_id and path: echo the exact values from the user message\n"
                    "- headers_hint: must be a list of {{name, value}} objects\n"
                    "- Do not invent file paths not provided in the allowed lists"
                ),
            },
            {
                "role": "user",
                "content": (
                    "artifact_id: {artifact_id} (echo this exactly)\n"
                    "path: {artifact_path} (echo this exactly)\n"
                    "kind: {kind}\n"
                    "purpose: {purpose}\n\n"
                    "Environment theme: {theme}\n"
                    "Reference pages: {reference_urls}\n"
                    "Reference notes: {reference_notes}\n\n"
                    "Paths you MUST reference (from artifact.links_to): {must_reference}\n"
                    "Additional paths you MAY reference (other bundle artifacts/assets): {may_reference}\n"
                    "Do not reference any paths outside these two lists.\n\n"
                    "Kind-specific guidance:\n{kind_directive}\n\n"
                    "--- Content Model Skeleton (replace ALL <placeholders> with concrete values) ---\n"
                    "{content_model_skeleton}\n"
                    "--- End Skeleton ---\n\n"
                    "Example output for html_page kind (for reference only):\n"
                    "{{\n"
                    "  \"artifact_id\": \"login-page\",\n"
                    "  \"path\": \"/admin/login.php\",\n"
                    "  \"kind\": \"html_page\",\n"
                    "  \"content_model\": {{\n"
                    "    \"title\": \"Administrative Login\",\n"
                    "    \"heading\": \"Sign In\",\n"
                    "    \"paragraphs\": [\"Enter your credentials to access the admin portal.\"],\n"
                    "    \"linked_stylesheets\": [\"/assets/admin.css\"],\n"
                    "    \"linked_scripts\": [\"/assets/login.js\"],\n"
                    "    \"nav_links\": [{{\"label\": \"Forgot Password\", \"href\": \"/admin/reset.php\"}}],\n"
                    "    \"form\": {{\n"
                    "      \"action\": \"/admin/login.php\",\n"
                    "      \"method\": \"post\",\n"
                    "      \"fields\": [\n"
                    "        {{\"name\": \"username\", \"label\": \"Username\", \"type\": \"text\"}},\n"
                    "        {{\"name\": \"password\", \"label\": \"Password\", \"type\": \"password\"}}\n"
                    "      ],\n"
                    "      \"submit_label\": \"Sign In\"\n"
                    "    }},\n"
                    "    \"footer\": \"Acme Corp Internal Systems\"\n"
                    "  }},\n"
                    "  \"headers_hint\": [],\n"
                    "  \"review_notes\": []\n"
                    "}}\n\n"
                    "Generate the artifact according to the purpose and kind-specific guidance."
                ).format(
                    artifact_id=artifact.artifact_id,
                    artifact_path=artifact.path,
                    kind=artifact.kind,
                    purpose=artifact.purpose,
                    theme=expert_spec.environment_theme,
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
            draft = heuristic_draft
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
            "trace_notes": ["coder:{}".format(draft.path)],
        }

    async def _assemble_bundle(self, state: GraphState):
        request = state["request"]
        resource_plan = state["resource_plan"]
        reference_pack = state.get("reference_pack") or ReferencePack()
        current_revision = state.get("plan_revision", 0)
        current_drafts = [
            draft for draft in state.get("artifact_drafts", []) if draft.plan_revision == current_revision
        ]
        if not current_drafts:
            raise ValidationError("no artifact drafts were produced for plan revision {}".format(current_revision))

        artifacts = self._generated_artifacts_from_reference_pack(reference_pack)
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
        reference_pack = state.get("reference_pack") or ReferencePack()
        reference_evidence = self._summarize_reference_pages(reference_pack.reference_pages)
        fetched_asset_summary = ", ".join(
            "{} ({})".format(asset.local_path, asset.kind) for asset in reference_pack.fetched_assets
        ) if reference_pack.fetched_assets else "none"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Review role in a honeypot generator evaluating content quality and realism.\n\n"
                    "Deterministic validators have already confirmed:\n"
                    "- Structural integrity: primary path exists, all internal references resolve\n"
                    "- Safety: no external asset links, no duplicate paths\n"
                    "- Budget compliance: artifact count and byte limits satisfied\n\n"
                    "Your role is to evaluate QUALITY defects the validators cannot detect.\n\n"
                    "You have veto power over these specific defect categories:\n"
                    "1. PLACEHOLDER_CONTENT: Artifact contains obvious placeholder text (lorem ipsum, 'example', 'test', 'TODO', '<placeholder>')\n"
                    "2. UNREALISTIC_FIELDS: Form field names are generic (field1, field2, input1) instead of realistic (username, password, email)\n"
                    "3. MINIMAL_CONFIG: Config files have fewer than 3 entries or use obviously fake values ('changeme', 'password123')\n"
                    "4. BROKEN_THEME: Content contradicts the environment_theme (e.g., 'WordPress' theme but page says 'Drupal Admin')\n"
                    "5. INTERNAL_LANGUAGE_LEAK: Text contains honeypot/lure/attacker/decoy/trap or synonyms despite the safety rule\n\n"
                    "Return 'revise' if you find ANY of these blocking defects. Specify the defect category and affected artifact in required_fixes.\n"
                    "Return 'approve' for minor style issues, missing CSS polish, or other non-blocking gaps.\n"
                    "Return 'fallback' only if multiple severe defects make the bundle unusable."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Intent family: {intent}\n"
                    "Environment theme: {theme}\n"
                    "Primary path: {primary}\n"
                    "Bundle paths: {paths}\n\n"
                    "Bundle evidence (structured features + content preview):\n{evidence}\n\n"
                    "Reference pages consulted: {reference_evidence}\n"
                    "Fetched assets: {fetched_assets}\n\n"
                    "Evaluate the bundle for the 5 quality defect categories listed in the system prompt.\n"
                    "A link to {index_page} is permitted even if that page is not in this bundle."
                ).format(
                    intent=expert_spec.intent_family,
                    theme=expert_spec.environment_theme,
                    primary=bundle.primary_path,
                    paths=", ".join(artifact.path for artifact in bundle.artifacts),
                    evidence=review_evidence,
                    reference_evidence=reference_evidence,
                    fetched_assets=fetched_asset_summary,
                    index_page=request.index_page,
                )
            },
        ]

        try:
            structured_decision = await self._invoke_structured("review", StructuredReviewDecision, messages)
            structured_decision = StructuredReviewDecision.model_validate(structured_decision)
            decision = self._normalize_review_decision(structured_decision)
        except Exception as error:
            self.logger.info("Falling back to deterministic review for %s: %s", request.normalized_path, error)
            decision = ReviewDecision(decision="approve", reasons=["deterministic validation passed"], required_fixes=[])

        # Let the LLM review verdict stand - it has veto power over quality defects
        if decision.decision != "approve":
            self.logger.info(
                "Review rejected bundle for %s with decision '%s': %s",
                request.normalized_path,
                decision.decision,
                decision.reasons or decision.required_fixes,
            )
        
        trace_decision = decision.decision if decision.decision == "approve" else "quality_defect:{}".format(decision.decision)
        return {"review_decision": decision, "trace_notes": ["review:{}".format(trace_decision)]}

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
                "review_summary": "; ".join(review_decision.reasons) if review_decision and review_decision.reasons else (
                    review_decision.decision if review_decision else bundle.review_summary
                ),
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
        required_kind = self._required_kind_for_path(request.normalized_path)
        default_primary_kind = required_kind or kind_by_intent[intent_family]
        references = research.references if research is not None else []
        supporting_context = research.snippets if research is not None and research.snippets else self._supporting_context(intent_family)

        return ExpertSpec(
            intent_family=intent_family,
            attacker_goal=goal_by_intent[intent_family],
            confidence=0.88,
            primary_resource_kind=default_primary_kind,
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

        required_kind = self._required_kind_for_path(primary_path)
        if required_kind == "xml_document":
            support_candidates = ["/WANCfgSCPD.xml", "/WANIPConnSCPD.xml"]
            support_paths = [candidate for candidate in support_candidates if candidate != primary_path]
            if not support_paths:
                support_paths = ["/device.xml"]
            artifacts = [
                PlannedArtifact(
                    artifact_id="primary-xml",
                    path=primary_path,
                    kind="xml_document",
                    purpose="Primary XML service descriptor for the requested endpoint",
                    links_to=support_paths[:2],
                ),
                PlannedArtifact(
                    artifact_id="support-xml-a",
                    path=support_paths[0],
                    kind="xml_document",
                    purpose="Related XML service description referenced by the primary descriptor",
                    links_to=[primary_path],
                ),
            ]
            if len(support_paths) > 1:
                artifacts.append(
                    PlannedArtifact(
                        artifact_id="support-xml-b",
                        path=support_paths[1],
                        kind="xml_document",
                        purpose="Secondary XML service description linked from gateway metadata",
                        links_to=[primary_path],
                    )
                )
        elif required_kind == "json_document":
            support_path = "/version" if primary_path != "/version" else "/info"
            artifacts = [
                PlannedArtifact(
                    artifact_id="primary-json",
                    path=primary_path,
                    kind="json_document",
                    purpose="Primary JSON API payload for the requested endpoint",
                    links_to=[support_path],
                ),
                PlannedArtifact(
                    artifact_id="support-text",
                    path=support_path,
                    kind="plain_text",
                    purpose="Adjacent plaintext service metadata endpoint",
                ),
            ]
        elif required_kind == "plain_text":
            support_path = "/version" if primary_path != "/version" else "/info"
            artifacts = [
                PlannedArtifact(
                    artifact_id="primary-text",
                    path=primary_path,
                    kind="plain_text",
                    purpose="Primary plaintext response for the requested endpoint",
                    links_to=[support_path],
                ),
                PlannedArtifact(
                    artifact_id="support-text",
                    path=support_path,
                    kind="plain_text",
                    purpose="Adjacent plaintext metadata endpoint",
                    links_to=[primary_path],
                ),
            ]
        elif required_kind == "binary_asset":
            artifacts = [
                PlannedArtifact(
                    artifact_id="primary-binary",
                    path=primary_path,
                    kind="binary_asset",
                    purpose="Primary binary asset for the requested endpoint",
                    links_to=["/index.html"],
                ),
                PlannedArtifact(
                    artifact_id="asset-index",
                    path="/index.html",
                    kind="html_page",
                    purpose="Landing page that references the requested binary asset",
                    links_to=[primary_path],
                ),
            ]
        elif required_kind == "robots_txt":
            artifacts = [
                PlannedArtifact(
                    artifact_id="primary-robots",
                    path=primary_path,
                    kind="robots_txt",
                    purpose="Primary crawler policy endpoint",
                    links_to=["/index.html"],
                ),
                PlannedArtifact(
                    artifact_id="robots-index",
                    path="/index.html",
                    kind="html_page",
                    purpose="Landing page that links to crawler policy",
                    links_to=[primary_path],
                ),
            ]
        elif required_kind == "sitemap_xml":
            artifacts = [
                PlannedArtifact(
                    artifact_id="primary-sitemap",
                    path=primary_path,
                    kind="sitemap_xml",
                    purpose="Primary sitemap endpoint for nearby resources",
                    links_to=["/robots.txt", "/index.html"],
                ),
                PlannedArtifact(
                    artifact_id="sitemap-robots",
                    path="/robots.txt",
                    kind="robots_txt",
                    purpose="Crawler policy associated with the sitemap",
                ),
                PlannedArtifact(
                    artifact_id="sitemap-index",
                    path="/index.html",
                    kind="html_page",
                    purpose="Landing page linking to sitemap and robots endpoints",
                    links_to=[primary_path, "/robots.txt"],
                ),
            ]
        elif intent_family == "config_theft":
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
        if intent_family == "cms_probe":
            return (
                "For WordPress-like login surfaces, include a local stylesheet artifact linked by login HTML pages. "
                "If you need logos/icons/fonts or other binary assets, either generate artifact.kind='binary_asset' (for synthetic bytes) or use reference_asset_plan.asset_fetches (for copied real assets)."
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
            "design_validation_iteration": 0,
            "design_validation_decision": "approve",
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
