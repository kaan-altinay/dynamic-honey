from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from tanner.generator.agentic.config import load_runtime_config
from tanner.generator.agentic.prompts import load_prompt_library
from tanner.generator.agentic.fallback import build_fallback_bundle
from tanner.generator.agentic.model_factory import build_role_model
from tanner.generator.agentic.models import (
    GeneratedBundle,
    GenerationRequest,
    GeneratorRuntimeConfig,
)
from tanner.generator.agentic.validators import (
    ensure_generation_request,
    validate_bundle,
)
from tanner.generator.agentic.state import GraphState
from tanner.generator.base_generator import BaseGenerator
from tanner.generator.agentic.roles.expert import ExpertRoleMixin
from tanner.generator.agentic.roles.design import DesignRoleMixin
from tanner.generator.agentic.roles.coder import CoderRoleMixin
from tanner.generator.agentic.roles.reviewer import ReviewerRoleMixin
from tanner.generator.agentic.roles.flow_designer import FlowDesignerRoleMixin
from tanner.generator.agentic.roles.shared import SharedRoleMixin


class AgenticBundleGenerator(
    ExpertRoleMixin,
    DesignRoleMixin,
    CoderRoleMixin,
    ReviewerRoleMixin,
    FlowDesignerRoleMixin,
    SharedRoleMixin,
    BaseGenerator,
):
    """Orchestrates the agentic LangGraph pipeline; node/role logic lives in roles/*.py mixins."""

    def __init__(self, runtime_config: GeneratorRuntimeConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.runtime_config = runtime_config or load_runtime_config()
        self.prompts = load_prompt_library()
        self._review_log_path = Path(self.runtime_config.review_log_path)
        self._role_models = {}
        self._invoke_loop = None
        self._invoke_semaphore = None
        self._invoke_spacing_lock = None
        self._next_model_call_time = 0.0
        self._graph_builder = self._build_graph()

    def _ensure_invoke_primitives(self) -> None:
        """Initialize loop-bound asyncio primitives for the active event loop."""
        loop = asyncio.get_running_loop()
        if self._invoke_loop is loop and self._invoke_semaphore is not None and self._invoke_spacing_lock is not None:
            return

        self._invoke_loop = loop
        self._invoke_semaphore = asyncio.Semaphore(self.runtime_config.max_concurrent_model_calls)
        self._invoke_spacing_lock = asyncio.Lock()
        self._next_model_call_time = 0.0

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
        if self.runtime_config.enable_scripted_flows:
            builder.add_node("flow_designer", self._flow_designer_node)
            builder.add_edge("assemble_bundle", "flow_designer")
            builder.add_edge("flow_designer", "review_node")
        else:
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

    @staticmethod
    def _diagnostic_event(
        stage: str,
        category: str,
        target: str,
        message: str,
        **details,
    ) -> dict:
        event = {
            "stage": stage,
            "category": category,
            "target": target,
            "message": message,
        }
        if details:
            event["details"] = details
        return event

    async def _wait_for_model_slot(self) -> None:
        self._ensure_invoke_primitives()

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
        self._ensure_invoke_primitives()

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
            # OpenAI strict response_format rejects some schemas we intentionally
            # use for V2 planning/generation:
            # - StructuredJsonDocumentDraft allows arbitrary JSON content_model.document
            # - ResourcePlan now carries first-class flow metadata and free-form
            #   coherence_facts, which produce a schema OpenAI rejects under strict
            #   response_format validation
            # Use function-calling for those schemas to preserve structured parsing
            # without weakening other roles unnecessarily.
            structured_output_method = None
            schema_name = getattr(schema, "__name__", "")
            if (
                (role_name == "coder" and schema_name == "StructuredJsonDocumentDraft")
                or (role_name == "design" and schema_name == "ResourcePlan")
            ):
                structured_output_method = "function_calling"

            if structured_output_method is None:
                runnable = model.with_structured_output(schema)
            else:
                runnable = model.with_structured_output(schema, method=structured_output_method)

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
            "generation_diagnostics": [],
        }

    @staticmethod
    async def _probe_internet_connectivity(timeout: float = 3.0) -> bool:
        """Return True if a basic TCP route to the internet is reachable."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("1.1.1.1", 53),
                timeout=timeout,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except Exception:
            return False

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
            "generation_diagnostics": [],
            "plan_revision": 0,
        }
        thread_id = "meta:{}:{}".format(request.normalized_path, uuid.uuid4())

        if self.runtime_config.enable_live_research:
            if not await self._probe_internet_connectivity():
                self.logger.warning(
                    "Internet connectivity check failed for %s — web research will be skipped "
                    "and LLM API calls may also fail. Check network connectivity on this host.",
                    request.normalized_path,
                )
        try:
            async with AsyncSqliteSaver.from_conn_string(self.runtime_config.checkpoint_path) as checkpointer:
                if not hasattr(checkpointer.conn, "is_alive"):
                    checkpointer.conn.is_alive = lambda: True
                graph = self._graph_builder.compile(checkpointer=checkpointer)
                result = await graph.ainvoke(
                    initial_state,
                    config={
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": self.runtime_config.graph_recursion_limit,
                    }
                )
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
