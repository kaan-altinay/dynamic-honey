from __future__ import annotations

import fcntl
import json
import re
from datetime import datetime, timezone

from tanner.generator.agentic.models import (
    FlowDescriptor,
    GeneratedBundle,
    GenerationRequest,
    ReferencePack,
    ReviewDecision,
    StructuredReviewDecision,
)
from tanner.generator.agentic.validators import (
    ValidationError,
    validate_bundle,
    validate_plan,
)
from tanner.generator.agentic.state import GraphState


class ReviewerRoleMixin:
    """Review role: evaluates a generated bundle for quality defects."""

    async def _review_node(self, state: GraphState):
        request = state["request"]
        expert_spec = state["expert_spec"]
        resource_plan = state["resource_plan"]
        bundle = GeneratedBundle.model_validate(state["generated_bundle"])
        issues = list(state.get("errors", []))
        hard_failures: list[str] = []
        flow_dict = state.get("flow_descriptor")
        if flow_dict is not None and bundle.flow_descriptor is None:
            try:
                bundle = bundle.model_copy(update={"flow_descriptor": FlowDescriptor.model_validate(flow_dict)})
            except Exception as error:
                message = "invalid flow descriptor: {}".format(error)
                issues.append(message)
                hard_failures.append(message)

        trim_diagnostics: list[dict] = []
        try:
            bundle, trim_diagnostics = self._fit_bundle_to_byte_limit(bundle, request)
            validate_plan(resource_plan, request, self.runtime_config)
            validate_bundle(bundle, request, self.runtime_config)
        except ValidationError as error:
            message = str(error)
            issues.append(message)
            hard_failures.append(message)

        if issues:
            return self._review_revise_or_fallback(
                state,
                issues,
                hard_failure=bool(hard_failures),
            )

        review_evidence = self._build_review_evidence(bundle)
        flow_evidence = self._build_flow_evidence(bundle)
        coherence_facts = json.dumps(resource_plan.coherence_facts, sort_keys=True) if resource_plan.coherence_facts else "none"
        reference_pack = state.get("reference_pack") or ReferencePack()
        reference_evidence = self._summarize_reference_pages(reference_pack.reference_pages)
        fetched_asset_summary = ", ".join(
            "{} ({})".format(asset.local_path, asset.kind) for asset in reference_pack.fetched_assets
        ) if reference_pack.fetched_assets else "none"
        enable_scripted_flows = self.runtime_config.enable_scripted_flows
        messages = [
            {
                "role": "system",
                "content": self.prompts.review.system(enable_scripted_flows),
            },
            {
                "role": "user",
                "content": self.prompts.review.user(enable_scripted_flows).format(
                    intent=expert_spec.intent_family,
                    theme=expert_spec.environment_theme,
                    primary=bundle.primary_path,
                    paths=", ".join(artifact.path for artifact in bundle.artifacts),
                    evidence=review_evidence,
                    reference_evidence=reference_evidence,
                    fetched_assets=fetched_asset_summary,
                    flow_evidence=flow_evidence,
                    coherence_facts=coherence_facts,
                    index_page=request.index_page,
                ),
            },
        ]

        reviewer_output = None
        review_source = "model"
        try:
            reviewer_output = await self._invoke_structured("review", StructuredReviewDecision, messages)
            structured_decision = StructuredReviewDecision.model_validate(reviewer_output)
            decision = self._normalize_review_decision(structured_decision)
        except Exception as error:
            self.logger.info("Falling back to deterministic review for %s: %s", request.normalized_path, error)
            review_source = "deterministic"
            decision = ReviewDecision(decision="approve", reasons=["deterministic validation passed"], required_fixes=[])
            reviewer_output = decision.model_dump(mode="json")

        self._append_review_log(
            request.normalized_path,
            reviewer_output,
            decision,
            source=review_source,
        )

        # Let the LLM review verdict stand - it has veto power over quality defects
        if decision.decision != "approve":
            self.logger.info(
                "Review rejected bundle for %s with decision '%s': %s",
                request.normalized_path,
                decision.decision,
                decision.reasons or decision.required_fixes,
            )

        trace_decision = decision.decision if decision.decision == "approve" else "quality_defect:{}".format(decision.decision)
        extra: dict = {
            "generated_bundle": bundle,
            "review_decision": decision,
            "trace_notes": ["review:{}".format(trace_decision)],
        }
        diagnostics = list(trim_diagnostics)
        if review_source == "deterministic":
            extra["errors"] = [
                "review deterministic fallback for {}: {} {}".format(
                    request.normalized_path,
                    "error" if "error" not in str(reviewer_output) else reviewer_output,
                    "(model call failed)",
                )
            ]
            diagnostics.append(
                self._diagnostic_event(
                    "review", "deterministic_fallback", request.normalized_path,
                    "model review call failed; deterministic approval applied",
                )
            )
        elif decision.decision != "approve":
            diagnostics.append(
                self._diagnostic_event(
                    "review", "quality_rejection", request.normalized_path,
                    "; ".join(decision.reasons[:3]),
                    required_fixes=decision.required_fixes[:3],
                )
            )
        if diagnostics:
            extra["generation_diagnostics"] = diagnostics
        return extra

    def _review_revise_or_fallback(
        self,
        state: GraphState,
        reasons: list[str],
        *,
        hard_failure: bool = False,
    ):
        next_iteration = state.get("review_iteration", 0) + 1
        if next_iteration >= self.runtime_config.max_review_loops:
            if hard_failure:
                retain_reason = (
                    "review loop budget exhausted after {} iteration(s); falling back because structural validation failed".format(
                        next_iteration
                    )
                )
                decision = ReviewDecision(decision="fallback", reasons=[retain_reason] + reasons, required_fixes=[])
                route = "fallback"
            else:
                retain_reason = (
                    "review loop budget exhausted after {} iteration(s); retaining latest generated artifacts".format(
                        next_iteration
                    )
                )
                decision = ReviewDecision(decision="approve", reasons=[retain_reason] + reasons, required_fixes=[])
                route = "approve"
        else:
            decision = ReviewDecision(decision="revise", reasons=reasons, required_fixes=reasons)
            route = "revise"

        request = state.get("request")
        endpoint = request.normalized_path if isinstance(request, GenerationRequest) else "<unknown>"
        self._append_review_log(
            endpoint,
            decision.model_dump(mode="json"),
            decision,
            source="review_loop",
        )
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
    def _build_flow_evidence(bundle: GeneratedBundle) -> str:
        """Summarize flow rules so review evaluates scripted behavior too."""
        descriptor = getattr(bundle, "flow_descriptor", None)
        if descriptor is None:
            return "none"

        lines = []
        for rule in sorted(descriptor.rules, key=lambda value: value.priority, reverse=True):
            cond = rule.condition
            cond_parts = []
            if cond is not None:
                if cond.method:
                    cond_parts.append("method={}".format(cond.method))
                if cond.requires_cookie:
                    cond_parts.append("requires_cookie={}".format(cond.requires_cookie))
                if cond.missing_cookie:
                    cond_parts.append("missing_cookie={}".format(cond.missing_cookie))
                if cond.requires_prev_path:
                    cond_parts.append("requires_prev_path={}".format(cond.requires_prev_path))
                if cond.min_post_count_to_path is not None:
                    cond_parts.append("min_post_count={}".format(cond.min_post_count_to_path))
                if cond.min_prior_post_count_to_path is not None:
                    cond_parts.append("min_prior_post_count={}".format(cond.min_prior_post_count_to_path))
                if cond.lockout_window_seconds is not None:
                    cond_parts.append("lockout_window_seconds={}".format(cond.lockout_window_seconds))
                if cond.lockout_active is not None:
                    cond_parts.append("lockout_active={}".format(cond.lockout_active))
                if cond.requires_header:
                    cond_parts.append("requires_header={}".format(cond.requires_header))
                if cond.missing_header:
                    cond_parts.append("missing_header={}".format(cond.missing_header))
                if cond.header_equals:
                    cond_parts.append("header_equals={}".format(cond.header_equals))
                if cond.header_contains:
                    cond_parts.append("header_contains={}".format(cond.header_contains))
                if cond.query_has:
                    cond_parts.append("query_has={}".format(cond.query_has))
                if cond.query_equals:
                    cond_parts.append("query_equals={}".format(cond.query_equals))
                if cond.query_contains:
                    cond_parts.append("query_contains={}".format(cond.query_contains))
                if cond.post_has:
                    cond_parts.append("post_has={}".format(cond.post_has))
                if cond.post_equals:
                    cond_parts.append("post_equals={}".format(cond.post_equals))
                if cond.post_contains:
                    cond_parts.append("post_contains={}".format(cond.post_contains))
            response = rule.response
            action_parts = []
            if response.artifact_path:
                action_parts.append("serve {}".format(response.artifact_path))
            if response.redirect_to:
                action_parts.append("redirect {}".format(response.redirect_to))
            if response.set_cookie:
                action_parts.append("set_cookie={}".format(sorted(response.set_cookie)))
            if response.clear_cookie:
                action_parts.append("clear_cookie={}".format(response.clear_cookie))
            lines.append(
                "priority={priority} match={match} when={condition} action={action}".format(
                    priority=rule.priority,
                    match=rule.match_path,
                    condition=", ".join(cond_parts) if cond_parts else "always",
                    action=", ".join(action_parts) if action_parts else "none",
                )
            )
        return "\n".join(lines) if lines else "none"

    @staticmethod
    def _normalize_review_decision(structured_decision: StructuredReviewDecision) -> ReviewDecision:
        return ReviewDecision(
            decision=structured_decision.decision,
            reasons=structured_decision.reasons or [],
            required_fixes=structured_decision.required_fixes or [],
        )

    @staticmethod
    def _serialize_reviewer_output(review_output):
        if hasattr(review_output, "model_dump"):
            try:
                review_output = review_output.model_dump(mode="json", exclude_none=False)
            except Exception:
                try:
                    review_output = review_output.model_dump()
                except Exception:
                    review_output = str(review_output)
        try:
            json.dumps(review_output)
            return review_output
        except TypeError:
            return str(review_output)

    def _append_review_log(
        self,
        endpoint: str,
        review_output,
        decision: ReviewDecision,
        source: str,
    ) -> None:
        timestamp = datetime.now(timezone.utc)
        entry = {
            "date": timestamp.date().isoformat(),
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "source": source,
            "review_output": self._serialize_reviewer_output(review_output),
            "decision": decision.model_dump(mode="json"),
        }

        try:
            self._review_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._review_log_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.seek(0)
                    raw_payload = handle.read().strip()
                    payload = json.loads(raw_payload) if raw_payload else {}
                    if not isinstance(payload, dict):
                        payload = {}

                    history = payload.get(endpoint)
                    if not isinstance(history, list):
                        history = []
                    history.append(entry)
                    payload[endpoint] = history

                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps(payload, indent=2, sort_keys=True))
                    handle.write("\n")
                    handle.flush()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception as error:
            self.logger.warning(
                "Unable to append review log at %s for %s: %s",
                self._review_log_path,
                endpoint,
                error,
            )
