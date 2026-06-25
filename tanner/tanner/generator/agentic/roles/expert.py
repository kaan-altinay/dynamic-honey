from __future__ import annotations

import asyncio
import re

from tanner.generator.agentic.models import (
    ExpertSpec,
    GenerationRequest,
)
from tanner.generator.agentic.tools import web_research
from tanner.generator.agentic.validators import (
    infer_intent_family,
)
from tanner.generator.agentic.state import GraphState


class ExpertRoleMixin:
    """Expert role: infers attacker intent from the requested path."""

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
        enable_scripted_flows = self.runtime_config.enable_scripted_flows
        messages = [
            {
                "role": "system",
                "content": self.prompts.expert.system(enable_scripted_flows),
            },
            {
                "role": "user",
                "content": self.prompts.expert.user(enable_scripted_flows).format(
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
            return {
                "expert_spec": expert_spec,
                "trace_notes": ["expert:heuristic:{}".format(expert_spec.intent_family)],
                "errors": ["expert fallback for {}: {} {}".format(request.normalized_path, error.__class__.__name__, error)],
                "generation_diagnostics": [
                    self._diagnostic_event(
                        "expert", "heuristic_fallback", request.normalized_path,
                        str(error), exception_type=error.__class__.__name__,
                    )
                ],
            }

        return {"expert_spec": expert_spec, "trace_notes": ["expert:{}".format(expert_spec.intent_family)]}

    async def _heuristic_expert_spec(self, request: GenerationRequest) -> ExpertSpec:
        intent_family = self._v2_intent_family(request.normalized_path) if self.runtime_config.enable_scripted_flows else infer_intent_family(request.normalized_path)
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
            "auth_portal": "Administrative authentication portal",
            "network_device": "Network device management interface",
            "container_api": "Container service API",
            "kubernetes_api": "Cluster orchestration API",
            "elastic_api": "Search cluster API",
            "solr_api": "Search administration API",
            "config_secret": "Production application secrets",
            "webshell_probe": "Web command or upload endpoint",
        }
        goal_by_intent = {
            "config_theft": "Obtain credentials, hostnames, and secret material from leaked configuration files.",
            "cms_probe": "Locate a believable CMS login surface and linked assets for credential capture.",
            "backup_probe": "Discover archived copies, export listings, or backup manifests with operational clues.",
            "admin_portal": "Reach an administrative login or dashboard surface with nearby supporting assets.",
            "framework_probe": "Verify the underlying stack and enumerate framework-specific resources.",
            "generic_recon": "Map the service and test whether nearby resources disclose useful context.",
            "auth_portal": "Reach an authentication or authorization surface and probe stateful access behavior.",
            "network_device": "Identify network device management functions, XML services, and nearby control paths.",
            "container_api": "Enumerate container/runtime API resources and related image or service metadata.",
            "kubernetes_api": "Enumerate cluster API resources with coherent namespaces, nodes, pods, and services.",
            "elastic_api": "Enumerate search cluster indices, nodes, health, and statistics.",
            "solr_api": "Enumerate search cores and administration metadata.",
            "config_secret": "Obtain credentials, hostnames, and secret material from leaked configuration files.",
            "webshell_probe": "Probe command/upload surfaces while receiving fixed safe response artifacts.",
        }
        kind_by_intent = {
            "config_theft": "config_text",
            "cms_probe": "html_page",
            "backup_probe": "backup_manifest",
            "admin_portal": "html_page",
            "framework_probe": "html_page",
            "generic_recon": "html_page",
            "auth_portal": "html_page",
            "network_device": "xml_document",
            "container_api": "json_document",
            "kubernetes_api": "json_document",
            "elastic_api": "json_document",
            "solr_api": "json_document",
            "config_secret": "config_text",
            "webshell_probe": "html_page",
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

    @staticmethod
    def _v2_intent_family(normalized_path: str) -> str:
        """Route V2-only endpoint families using broad path semantics."""
        lowered = normalized_path.lower()
        tokens = set(re.split(r"[^a-z0-9]+", lowered))
        tokens.discard("")

        if lowered.endswith(".env") or lowered in {"/.env", "/api/.env", "/env/.env", "/.git/config"}:
            return "config_secret"
        if lowered.startswith("/solr/") or "solr" in tokens:
            return "solr_api"
        if lowered.startswith("/api/v1/") or tokens.intersection({"pods", "services", "namespaces"}):
            return "kubernetes_api"
        if lowered.startswith("/_cat/") or lowered in {"/_nodes", "/_nodes/_local", "/_stats"} or tokens.intersection({"indices", "stats"}):
            return "elastic_api"
        if lowered in {"/containers/json", "/json/version", "/v2/_catalog"} or tokens.intersection({"containers", "catalog"}):
            return "container_api"
        if lowered in {"/hnap1", "/tr064dev.xml", "/igd.xml", "/wsman"} or tokens.intersection({"hnap1", "tr064dev", "igd", "wsman", "boaform", "setup"}):
            return "network_device"
        if tokens.intersection({"shell", "upload", "download", "powershell"}) or re.search(r"/(?:upl|get|1)\.php$", lowered):
            return "webshell_probe"
        if tokens.intersection({"login", "logon", "auth", "signin", "password", "sslvpn", "remote", "manager"}):
            return "auth_portal"
        return infer_intent_family(normalized_path)

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
