from __future__ import annotations

import asyncio
import re

from tanner.generator.agentic.models import (
    EndpointSemanticHint,
    ExpertSpec,
    GenerationRequest,
    PlannedArtifact,
    ReferencePage,
    ResourcePlan,
    ReviewDecision,
)
from tanner.generator.agentic.tools import fetch_reference_page, web_research
from tanner.generator.agentic.validators import (
    ValidationError,
    normalize_path,
    validate_plan,
)
from tanner.generator.agentic.state import GraphState


class DesignRoleMixin:
    """Design role: plans the ResourcePlan (artifact bundle layout) for a request."""

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
        enable_scripted_flows = self.runtime_config.enable_scripted_flows
        semantic_hint_text = (
            self._format_semantic_hint(self._endpoint_semantic_hint(request.normalized_path))
            if enable_scripted_flows
            else ""
        )

        def _build_design_messages(reference_candidates: str):
            return [
                {
                    "role": "system",
                    "content": self.prompts.design.system(enable_scripted_flows).format(
                        max_artifacts=self.runtime_config.max_bundle_artifacts,
                    ),
                },
                {
                    "role": "user",
                    "content": self.prompts.design.user(enable_scripted_flows).format(
                        path=request.normalized_path,
                        intent=expert_spec.intent_family,
                        goal=expert_spec.attacker_goal,
                        feedback=", ".join(review_feedback) if review_feedback else "none",
                        count=self.runtime_config.max_bundle_artifacts,
                        bytes=self.runtime_config.max_bundle_bytes,
                        rule=self._design_guardrails_for_intent(expert_spec.intent_family),
                        reference_summary=reference_candidates,
                        semantic_hint_text=semantic_hint_text,
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
                "trace_notes": ["design:heuristic:{}:{}".format(resource_plan.primary_path, len(resource_plan.artifacts))],
                "errors": ["design fallback for {}: {} {}".format(request.normalized_path, last_error.__class__.__name__ if last_error else "unknown", last_error)],
                "generation_diagnostics": [
                    self._diagnostic_event(
                        "design", "heuristic_fallback", request.normalized_path,
                        str(last_error) if last_error else "all design attempts failed",
                        exception_type=last_error.__class__.__name__ if last_error else "unknown",
                        attempts=len(attempt_payloads),
                    )
                ],
            }

        return {
            "resource_plan": resource_plan,
            "review_feedback": [],
            "plan_revision": plan_revision,
            "trace_notes": ["design:{}:{}".format(resource_plan.primary_path, len(resource_plan.artifacts))],
        }

    def _design_revise_or_fallback(self, state: GraphState, reasons: list[str]):
        next_iteration = state.get("design_validation_iteration", 0) + 1
        max_design_validation_loops = getattr(
            self.runtime_config,
            "max_design_validation_loops",
            self.runtime_config.max_review_loops,
        )
        if next_iteration >= max_design_validation_loops:
            # Budget exhausted.  Rather than producing a deterministic fallback
            # stub, substitute the heuristic plan (which always passes gate
            # validation) and proceed through the normal reference→coder→review
            # path so that real LLM-generated content is persisted.
            request = state.get("request")
            expert_spec = state.get("expert_spec")
            if request is None:
                # Defensive guard: request missing from state — should not occur in
                # normal operation but can happen in isolated unit tests.  Fall
                # through to the legacy fallback path so no AttributeError is raised.
                return {
                    "design_validation_decision": "fallback",
                    "design_validation_iteration": next_iteration,
                    "review_decision": ReviewDecision(decision="fallback", reasons=reasons, required_fixes=reasons),
                    "review_feedback": reasons,
                    "trace_notes": ["design_gate:budget_exhausted:no_request_in_state"],
                }
            heuristic_plan = self._heuristic_plan(
                request, expert_spec, state.get("review_feedback", [])
            )
            return {
                "design_validation_decision": "approve",
                "design_validation_iteration": next_iteration,
                "resource_plan": heuristic_plan,
                "review_feedback": reasons,
                "trace_notes": [
                    "design_gate:budget_exhausted:heuristic_fallthrough:{}".format(
                        heuristic_plan.primary_path
                    )
                ],
            }
        return {
            "design_validation_decision": "revise",
            "design_validation_iteration": next_iteration,
            "review_decision": ReviewDecision(decision="revise", reasons=reasons, required_fixes=reasons),
            "review_feedback": reasons,
            "trace_notes": ["design_gate:revise"],
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
            flow_response = artifact.flow_response
            if flow_response is not None:
                response_updates = {}
                if flow_response.artifact_path:
                    response_updates["artifact_path"] = normalize_path(flow_response.artifact_path, index_page=request.index_page)
                if flow_response.redirect_to:
                    response_updates["redirect_to"] = normalize_path(flow_response.redirect_to, index_page=request.index_page)
                if response_updates:
                    flow_response = flow_response.model_copy(update=response_updates)
            normalized_artifacts.append(
                artifact.model_copy(
                    update={
                        "path": normalize_path(artifact.path, index_page=request.index_page),
                        "links_to": normalized_links,
                        "flow_match_path": normalize_path(artifact.flow_match_path, index_page=request.index_page) if artifact.flow_match_path else None,
                        "flow_response": flow_response,
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
    def _design_guardrails_for_intent(intent_family: str) -> str:
        if intent_family == "config_theft":
            return (
                "Include the primary leaked configuration file plus at least one adjacent supporting artifact such as a log excerpt, "
                "backup manifest, or alternate config artifact. Never use internal planning words in served content."
            )
        if intent_family == "cms_probe":
            return (
                "For WordPress-like login surfaces, include a local stylesheet artifact linked by login HTML pages. "
                "If you need logos/icons/fonts or other binary assets, either generate artifact.kind='binary_asset' (for synthetic bytes) or use reference_asset_plan.asset_fetches (for copied real assets)."
            )
        if intent_family == "config_secret":
            return (
                "Include the primary leaked configuration/secret artifact plus at least one adjacent supporting artifact "
                "such as server status, debug status, robots, backup manifest, or related config. Keep values coherent."
            )
        if intent_family == "auth_portal":
            return (
                "Plan an authentication or authorization surface. Use first-class flow artifacts for auth challenges, "
                "session-expired states, or credential POST behavior only where the endpoint shape supports it."
            )
        if intent_family == "network_device":
            return (
                "Plan protocol-shaped network device resources with coherent device, firmware, service, and control-path facts. "
                "Prefer XML or service-description artifacts when the requested path suggests protocol discovery."
            )
        if intent_family in {"container_api", "kubernetes_api", "elastic_api", "solr_api"}:
            return (
                "Plan parseable API artifacts with shared coherence_facts for resource names, IDs, versions, timestamps, and related paths. "
                "Prefer coherent list/detail support artifacts over isolated responses."
            )
        if intent_family == "webshell_probe":
            return (
                "Represent command/upload probing with fixed safe response artifacts and request-shape flow conditions where useful. "
                "Never execute, interpolate, or reflect externally supplied commands."
            )
        return "Keep the bundle compact, coherent, and free of internal planning language."

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

        for artifact in artifacts:
            if artifact.response_contract.content_type is None:
                artifact.response_contract.content_type = self._default_content_type_for_artifact(
                    artifact.kind,
                    artifact.path,
                )
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

    @staticmethod
    def _endpoint_semantic_hint(path: str) -> EndpointSemanticHint:
        """Infer generic path semantics for V2 prompts without binding behavior to endpoints."""
        lowered = path.lower()
        raw_tokens = re.split(r"[^a-z0-9]+", lowered)
        tokens = [token for token in raw_tokens if token]
        token_set = set(tokens)
        resource_terms = [
            token
            for token in tokens
            if token in {
                "api", "v1", "v2", "json", "catalog", "container", "containers",
                "pod", "pods", "service", "services", "node", "nodes", "target", "targets",
                "db", "dbs", "index", "indices", "stats", "status", "config", "env",
                "debug", "backup", "version", "manager", "admin", "login", "auth",
                "session", "shell", "cmd", "exec", "upload", "download",
            }
        ]
        protocol_terms = [
            token
            for token in tokens
            if token in {"xml", "soap", "ws", "wsman", "cgi", "php", "json", "rest", "rpc"}
        ]

        interaction_styles = []
        response_shapes = []
        if token_set.intersection({"login", "auth", "signin", "password", "session", "manager", "admin"}):
            interaction_styles.append("credential form or authorization challenge")
            response_shapes.extend(["html", "redirect", "401"])
        if token_set.intersection({"api", "v1", "v2", "json", "catalog", "containers", "pods", "services", "nodes", "targets", "version"}):
            interaction_styles.append("API/resource discovery with coherent follow-up resources")
            response_shapes.append("json")
        if token_set.intersection({"config", "env", "settings", "server", "debug", "status"}):
            interaction_styles.append("configuration/status disclosure with reinforcing support paths")
            response_shapes.extend(["plain_text", "json"])
        if token_set.intersection({"xml", "soap", "wsman", "ws", "rpc"}):
            interaction_styles.append("XML or SOAP-like service description/action response")
            response_shapes.append("xml")
        if token_set.intersection({"shell", "cmd", "exec", "upload", "download", "cgi", "php"}):
            interaction_styles.append("command/upload probe with fixed safe response artifacts")
            response_shapes.extend(["plain_text", "html"])

        return EndpointSemanticHint(
            path_tokens=tokens,
            likely_resource_terms=resource_terms,
            likely_protocol_terms=protocol_terms,
            likely_interaction_styles=interaction_styles,
            suggested_response_shapes=sorted(set(response_shapes)),
        )

    @staticmethod
    def _format_semantic_hint(hint: EndpointSemanticHint) -> str:
        return (
            "path_tokens={tokens}; resource_terms={resources}; protocol_terms={protocols}; "
            "interaction_styles={styles}; suggested_response_shapes={shapes}"
        ).format(
            tokens=", ".join(hint.path_tokens) if hint.path_tokens else "none",
            resources=", ".join(hint.likely_resource_terms) if hint.likely_resource_terms else "none",
            protocols=", ".join(hint.likely_protocol_terms) if hint.likely_protocol_terms else "none",
            styles=", ".join(hint.likely_interaction_styles) if hint.likely_interaction_styles else "none",
            shapes=", ".join(hint.suggested_response_shapes) if hint.suggested_response_shapes else "none",
        )
