from __future__ import annotations

from typing import Any

from tanner.config import TannerConfig
from tanner.generator.agentic.models import GeneratorRoleConfig, GeneratorRuntimeConfig

_ROLE_NAMES = ("expert", "design", "coder", "review")


def _config_value(key: str, default: Any) -> Any:
    try:
        value = TannerConfig.get("GENERATOR", key)
    except KeyError:
        return default
    return default if value is None else value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_runtime_config() -> GeneratorRuntimeConfig:
    enable_scripted_flows = bool(_config_value("enable_scripted_flows", False))
    v2_overrides = _as_dict(_config_value("v2_overrides", {})) if enable_scripted_flows else {}

    def merged_value(key: str, default: Any) -> Any:
        if enable_scripted_flows and key in v2_overrides:
            value = v2_overrides.get(key)
            return default if value is None else value
        return _config_value(key, default)

    base_role_defaults = _as_dict(_config_value("role_defaults", {}))
    override_role_defaults = _as_dict(v2_overrides.get("role_defaults")) if enable_scripted_flows else {}
    role_defaults = dict(base_role_defaults)
    role_defaults.update(override_role_defaults)

    base_roles = _as_dict(_config_value("roles", {}))
    override_roles = _as_dict(v2_overrides.get("roles")) if enable_scripted_flows else {}
    configured_roles = {key: _as_dict(value) for key, value in base_roles.items()}
    for role_name, role_config in override_roles.items():
        merged_role = dict(_as_dict(base_roles.get(role_name)))
        merged_role.update(_as_dict(role_config))
        configured_roles[role_name] = merged_role

    roles = {}
    for role_name in _ROLE_NAMES:
        merged = dict(role_defaults)
        merged.update(_as_dict(configured_roles.get(role_name)))
        roles[role_name] = GeneratorRoleConfig.model_validate(merged)

    return GeneratorRuntimeConfig(
        backend=str(merged_value("backend", "agentic")),
        max_review_loops=int(merged_value("max_review_loops", 2)),
        max_design_validation_loops=int(merged_value("max_design_validation_loops", 2)),
        allow_fallback_persistence=bool(merged_value("allow_fallback_persistence", False)),
        max_bundle_artifacts=int(merged_value("max_bundle_artifacts", 4)),
        max_bundle_bytes=int(merged_value("max_bundle_bytes", 262_144)),
        checkpoint_path=str(merged_value("checkpoint_path", "/tmp/tanner-agentic-checkpoints.sqlite")),
        graph_recursion_limit=int(merged_value("graph_recursion_limit", 200)),
        review_log_path=str(merged_value("review_log_path", "/tmp/tanner-agentic-review-log.json")),
        enable_live_research=bool(merged_value("enable_live_research", True)),
        max_tool_response_chars=int(merged_value("max_tool_response_chars", 4_000)),
        max_command_output_chars=int(merged_value("max_command_output_chars", 4_000)),
        command_timeout=int(merged_value("command_timeout", 5)),
        max_concurrent_model_calls=int(merged_value("max_concurrent_model_calls", 4)),
        inter_call_delay_seconds=float(merged_value("inter_call_delay_seconds", 0.0)),
        max_rate_limit_retries=int(merged_value("max_rate_limit_retries", 2)),
        default_rate_limit_backoff_seconds=float(merged_value("default_rate_limit_backoff_seconds", 12.0)),
        max_length_limit_retries=int(merged_value("max_length_limit_retries", 2)),
        length_retry_token_increase=int(merged_value("length_retry_token_increase", 800)),
        max_length_retry_tokens=int(merged_value("max_length_retry_tokens", 6000)),
        enable_scripted_flows=enable_scripted_flows,
        roles=roles,
    )
