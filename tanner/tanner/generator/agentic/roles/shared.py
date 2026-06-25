from __future__ import annotations

import asyncio

from tanner.generator.agentic.fallback import build_fallback_bundle
from tanner.generator.agentic.models import (
    FlowDescriptor,
    ArtifactReferenceContext,
    GeneratedArtifact,
    GeneratedBundle,
    GenerationRequest,
    ReferencePack,
)
from tanner.generator.agentic.renderers import render_artifact
from tanner.generator.agentic.tools import fetch_reference_page, fetch_static_asset
from tanner.generator.agentic.validators import (
    ValidationError,
    _is_external_reference,
    diagnose_flow_reachability,
    extract_css_references,
    extract_html_references,
    extract_javascript_references,
    normalize_path,
    validate_artifact_draft,
    validate_bundle,
)
from tanner.generator.agentic.state import GraphState


class SharedRoleMixin:
    """Cross-role plumbing: reference-pack handling and bundle assembly/finalization."""

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

    @staticmethod
    def _bundle_bytes(artifacts: list[GeneratedArtifact]) -> int:
        return sum(len(artifact.body_bytes) for artifact in artifacts)

    def _referenced_bundle_paths(
        self,
        artifact: GeneratedArtifact,
        *,
        request: GenerationRequest,
        artifact_paths: set[str],
        flow_descriptor: FlowDescriptor | None,
    ) -> set[str]:
        if artifact.kind == "html_page":
            raw_references = extract_html_references(artifact.body_bytes)
        elif artifact.kind == "stylesheet":
            raw_references = extract_css_references(artifact.body_bytes)
        elif artifact.kind == "javascript":
            raw_references = extract_javascript_references(artifact.body_bytes)
        else:
            raw_references = []

        referenced_paths: set[str] = set()
        for reference in raw_references:
            candidate = reference.strip()
            if not candidate or _is_external_reference(candidate):
                continue
            normalized = normalize_path(candidate, index_page=request.index_page)
            if normalized in artifact_paths:
                referenced_paths.add(normalized)

        if flow_descriptor is None:
            return referenced_paths

        for rule in flow_descriptor.rules:
            if rule.match_path != artifact.path:
                continue
            if rule.response.artifact_path in artifact_paths:
                referenced_paths.add(rule.response.artifact_path)
            redirect_to = rule.response.redirect_to
            if redirect_to is None:
                continue
            normalized_redirect = normalize_path(redirect_to, index_page=request.index_page)
            if normalized_redirect in artifact_paths:
                referenced_paths.add(normalized_redirect)
        return referenced_paths

    def _fit_bundle_to_byte_limit(
        self,
        bundle: GeneratedBundle,
        request: GenerationRequest,
    ) -> tuple[GeneratedBundle, list[dict]]:
        original_artifact_count = len(bundle.artifacts)
        original_total_bytes = self._bundle_bytes(bundle.artifacts)
        if original_total_bytes <= self.runtime_config.max_bundle_bytes:
            return bundle, []

        artifact_by_path: dict[str, GeneratedArtifact] = {}
        for artifact in bundle.artifacts:
            if artifact.path in artifact_by_path:
                return bundle, []
            artifact_by_path[artifact.path] = artifact

        artifact_paths = set(artifact_by_path)
        flow_descriptor = bundle.flow_descriptor
        dependency_cache: dict[str, set[str]] = {}

        def dependency_closure(seed_paths: set[str]) -> set[str]:
            selected_paths: set[str] = set()
            pending_paths = [path for path in seed_paths if path in artifact_by_path]
            while pending_paths:
                path = pending_paths.pop()
                if path in selected_paths:
                    continue
                selected_paths.add(path)
                dependencies = dependency_cache.get(path)
                if dependencies is None:
                    dependencies = self._referenced_bundle_paths(
                        artifact_by_path[path],
                        request=request,
                        artifact_paths=artifact_paths,
                        flow_descriptor=flow_descriptor,
                    )
                    dependency_cache[path] = dependencies
                for dependency in dependencies:
                    if dependency not in selected_paths:
                        pending_paths.append(dependency)
            return selected_paths

        selected_paths = dependency_closure({bundle.primary_path})
        selected_total_bytes = sum(len(artifact_by_path[path].body_bytes) for path in selected_paths)
        if selected_total_bytes > self.runtime_config.max_bundle_bytes:
            return bundle, []

        flow_paths: set[str] = set()
        if flow_descriptor is not None:
            for rule in flow_descriptor.rules:
                if rule.match_path in artifact_by_path:
                    flow_paths.add(rule.match_path)
                if rule.response.artifact_path in artifact_paths:
                    flow_paths.add(rule.response.artifact_path)
                redirect_to = rule.response.redirect_to
                if redirect_to is None:
                    continue
                normalized_redirect = normalize_path(redirect_to, index_page=request.index_page)
                if normalized_redirect in artifact_by_path:
                    flow_paths.add(normalized_redirect)

        def artifact_priority(artifact: GeneratedArtifact) -> tuple[bool, bool, bool, bool, int, str]:
            return (
                artifact.path != bundle.primary_path,
                artifact.path not in flow_paths,
                artifact.artifact_scope != "dynamic_endpoint",
                artifact.kind in {"robots_txt", "sitemap_xml"},
                len(artifact.body_bytes),
                artifact.path,
            )

        remaining_artifacts = sorted(
            (artifact for artifact in bundle.artifacts if artifact.path not in selected_paths),
            key=artifact_priority,
        )
        for artifact in remaining_artifacts:
            additional_paths = dependency_closure({artifact.path}) - selected_paths
            if not additional_paths:
                continue
            additional_total_bytes = sum(
                len(artifact_by_path[path].body_bytes) for path in additional_paths
            )
            if selected_total_bytes + additional_total_bytes > self.runtime_config.max_bundle_bytes:
                continue
            selected_paths.update(additional_paths)
            selected_total_bytes += additional_total_bytes

        if len(selected_paths) == original_artifact_count:
            return bundle, []

        trimmed_bundle = bundle.model_copy(
            update={
                "artifacts": [artifact for artifact in bundle.artifacts if artifact.path in selected_paths],
            }
        )
        validate_bundle(trimmed_bundle, request, self.runtime_config)
        dropped_paths = [
            artifact.path for artifact in bundle.artifacts if artifact.path not in selected_paths
        ]
        diagnostics = [
            self._diagnostic_event(
                "bundle",
                "trimmed_to_byte_limit",
                request.normalized_path,
                "trimmed generated bundle to satisfy byte limit",
                original_artifact_count=original_artifact_count,
                final_artifact_count=len(trimmed_bundle.artifacts),
                original_total_bytes=original_total_bytes,
                final_total_bytes=selected_total_bytes,
                dropped_paths=dropped_paths,
            )
        ]
        return trimmed_bundle, diagnostics

    async def _prepare_reference_pack(self, state: GraphState):
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

    async def _finalize_bundle(self, state: GraphState):
        request = state["request"]
        bundle = GeneratedBundle.model_validate(state["generated_bundle"])
        review_decision = state.get("review_decision")
        artifacts = sorted(
            bundle.artifacts,
            key=lambda artifact: (artifact.path != bundle.primary_path, artifact.path),
        )
        # Carry V2 flow descriptor from flow_designer, preserving the bundle copy reviewed above.
        flow_descriptor = bundle.flow_descriptor
        flow_dict = state.get("flow_descriptor")
        if flow_dict is not None:
            try:
                flow_descriptor = FlowDescriptor.model_validate(flow_dict)
            except Exception:
                self.logger.warning("_finalize_bundle: invalid flow_descriptor in state; preserving reviewed bundle descriptor")
        finalized = bundle.model_copy(
            update={
                "artifacts": artifacts,
                "review_summary": "; ".join(review_decision.reasons) if review_decision and review_decision.reasons else (
                    review_decision.decision if review_decision else bundle.review_summary
                ),
                "flow_descriptor": flow_descriptor,
                "generation_trace": list(state.get("trace_notes", [])) + ["finalized"],
                "generation_errors": list(state.get("errors", [])),
                "generation_diagnostics": list(state.get("generation_diagnostics", [])),
            }
        )
        finalized, trim_diagnostics = self._fit_bundle_to_byte_limit(finalized, request)
        if trim_diagnostics:
            finalized = finalized.model_copy(
                update={
                    "generation_diagnostics": finalized.generation_diagnostics + trim_diagnostics,
                }
            )
        validate_bundle(finalized, request, self.runtime_config)
        flow_reachability_diagnostics = diagnose_flow_reachability(finalized)
        if flow_reachability_diagnostics:
            diagnostic_summary = "; ".join(
                "FLOW_REACHABILITY_WARNING: {}".format(diagnostic)
                for diagnostic in flow_reachability_diagnostics
            )
            finalized = finalized.model_copy(
                update={
                    "review_summary": "{}; {}".format(finalized.review_summary, diagnostic_summary)
                    if finalized.review_summary
                    else diagnostic_summary,
                }
            )
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
        bundle = bundle.model_copy(
            update={
                "generation_trace": list(state.get("trace_notes", [])) + ["fallback"],
                "generation_errors": list(state.get("errors", [])),
                "generation_diagnostics": list(state.get("generation_diagnostics", [])) + [
                    self._diagnostic_event(
                        "fallback_node", "final_fallback", request.normalized_path,
                        "; ".join(reasons[:5]),
                    )
                ],
            }
        )
        validate_bundle(bundle, request, self.runtime_config)
        return {"generated_bundle": bundle, "trace_notes": ["fallback"]}

    @staticmethod
    def _supporting_context(intent_family: str) -> list[str]:
        defaults = {
            "config_theft": [
                "Expose realistic database, cache, and mail settings.",
                "Adjacent files should imply a live production environment rather than a toy sample.",
            ],
            "cms_probe": [
                "The primary page should look like a believable CMS/admin entry surface with familiar assets.",
                "For login-like forms, include a stateful flow plan (failed-attempt and lockout/redirect-style outcomes) without implementing real authentication.",
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
                "The bundle should give the visitor enough context to keep exploring.",
                "Every referenced internal path must exist in the generated bundle.",
            ],
            "auth_portal": [
                "The bundle should expose a believable authentication or authorization surface.",
                "Use stateful flow behavior only when request shape or credential form behavior makes it plausible.",
            ],
            "network_device": [
                "The bundle should resemble a management or service-description surface for a network appliance.",
                "XML/control paths and status pages should share device names, firmware versions, and service labels.",
            ],
            "container_api": [
                "The bundle should present coherent container/runtime or registry resources.",
                "Repository, image, container, and version names should remain consistent across artifacts.",
            ],
            "kubernetes_api": [
                "The bundle should present coherent cluster API resources.",
                "Namespaces, nodes, pods, services, UIDs, and timestamps should remain consistent across artifacts.",
            ],
            "elastic_api": [
                "The bundle should present coherent search cluster resources.",
                "Cluster names, node IDs, index names, health values, and counts should align across artifacts.",
            ],
            "solr_api": [
                "The bundle should present coherent search administration resources.",
                "Core names, versions, handlers, and status fields should align across artifacts.",
            ],
            "config_secret": [
                "Expose realistic database, cache, mail, token, and internal URL settings.",
                "Supporting files should reinforce that the configuration came from a live deployment.",
            ],
            "webshell_probe": [
                "The bundle should safely respond to command/upload probes with fixed generated artifacts only.",
                "Never execute or reflect externally supplied commands; serve deterministic content.",
            ],
        }
        return defaults[intent_family]

    @staticmethod
    def _lure_requirements(intent_family: str) -> list[str]:
        requirements = {
            "config_theft": [
                "Expose realistic secret-looking values and infrastructure hostnames.",
                "Keep the file compact and internally consistent, as if pulled from a real deployment.",
            ],
            "cms_probe": [
                "Match common CMS login terminology and page structure with realistic field names.",
                "When forms submit with POST on login-like endpoints, include a stateful flow plan so repeated attempts do not always re-serve the same base page.",
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
                "Use deterministic static artifacts for baseline endpoints; only add stateful behavior where authentication-like form interactions make it plausible.",
            ],
            "generic_recon": [
                "Provide a believable static entry point for further browsing.",
                "Add at least one adjacent support artifact so the bundle feels contextual.",
            ],
            "auth_portal": [
                "Use realistic labels, field names, authorization challenges, and session-state variants where appropriate.",
                "Do not apply lockout behavior unless a credential POST form exists.",
            ],
            "network_device": [
                "Use protocol-shaped XML/HTML/status content with coherent device and firmware details.",
                "Advertised control/action paths should exist inside the bundle when budget allows.",
            ],
            "container_api": [
                "Use parseable JSON and shared repository/image/container/version facts.",
                "Related list/detail artifacts should refer to the same objects.",
            ],
            "kubernetes_api": [
                "Use parseable API-style JSON and shared cluster resource facts.",
                "Related artifacts should preserve namespace, node, pod, service, UID, and timestamp consistency.",
            ],
            "elastic_api": [
                "Use parseable JSON or text table artifacts with shared cluster/index/node facts.",
                "Related artifacts should preserve health, counts, and resource naming consistency.",
            ],
            "solr_api": [
                "Use parseable administration JSON/XML/text artifacts with shared core and handler facts.",
                "Related artifacts should preserve version, core, and status consistency.",
            ],
            "config_secret": [
                "Expose realistic secret-looking values and infrastructure hostnames.",
                "Add at least one adjacent support artifact so the leak feels contextual.",
            ],
            "webshell_probe": [
                "Represent command/upload behavior with fixed safe response artifacts.",
                "Never execute, interpolate, or reflect externally supplied commands.",
            ],
        }
        return requirements[intent_family]

    @staticmethod
    def _heuristic_page_copy(intent_family: str) -> list[str]:
        """Generic visible paragraph text for the no-LLM fallback path.

        Unlike _lure_requirements/_supporting_context (internal planning hints
        for the LLM-driven path, never meant to reach a rendered page), these
        strings are the actual paragraph text served on a heuristic html_page
        artifact. They must read as ordinary production copy with no planning
        or meta language.
        """
        copy_by_intent = {
            "config_theft": [
                "This system manages application configuration and deployment settings.",
                "Access is restricted to authorized operations personnel.",
            ],
            "cms_probe": [
                "Please sign in with your account credentials to continue.",
                "Contact your site administrator if you have trouble accessing your account.",
            ],
            "backup_probe": [
                "This page lists recent backup archives and export jobs for the platform.",
                "Retention and recovery are managed by the operations team.",
            ],
            "admin_portal": [
                "This is a restricted administrative area for authorized staff only.",
                "Sign in below to continue to the dashboard.",
            ],
            "framework_probe": [
                "This application is running on its configured web framework.",
                "Refer to the project documentation for available routes and modules.",
            ],
            "generic_recon": [
                "Welcome. This page provides general information about the service.",
                "Additional resources are linked below.",
            ],
            "auth_portal": [
                "Sign in to continue to your account.",
                "Sessions expire automatically after a period of inactivity.",
            ],
            "network_device": [
                "This is the management interface for the network appliance.",
                "Use the navigation below to review device and service status.",
            ],
            "container_api": [
                "This endpoint returns runtime information for the container service.",
                "See the API reference for available operations.",
            ],
            "kubernetes_api": [
                "This endpoint returns cluster resource information.",
                "Refer to the API documentation for supported resource types.",
            ],
            "elastic_api": [
                "This endpoint returns search cluster status and statistics.",
                "Refer to the cluster documentation for additional detail.",
            ],
            "solr_api": [
                "This endpoint returns search core and handler status.",
                "Refer to the administration guide for configuration options.",
            ],
            "config_secret": [
                "This page lists environment and service configuration values used by the application.",
                "Access is limited to internal systems.",
            ],
            "webshell_probe": [
                "This endpoint accepts diagnostic requests for the service.",
                "Invalid or malformed requests receive a generic error response.",
            ],
        }
        return copy_by_intent.get(
            intent_family,
            [
                "This page provides information about the service.",
                "Additional resources may be available via navigation links.",
            ],
        )
