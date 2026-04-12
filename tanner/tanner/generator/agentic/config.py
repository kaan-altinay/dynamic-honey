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
    role_defaults = _as_dict(_config_value("role_defaults", {}))
    configured_roles = _as_dict(_config_value("roles", {}))

    roles = {}
    for role_name in _ROLE_NAMES:
        merged = dict(role_defaults)
        merged.update(_as_dict(configured_roles.get(role_name)))
        roles[role_name] = GeneratorRoleConfig.model_validate(merged)

    return GeneratorRuntimeConfig(
        backend=str(_config_value("backend", "agentic")),
        max_review_loops=int(_config_value("max_review_loops", 2)),
        max_bundle_artifacts=int(_config_value("max_bundle_artifacts", 4)),
        max_bundle_bytes=int(_config_value("max_bundle_bytes", 262_144)),
        checkpoint_path=str(_config_value("checkpoint_path", "/tmp/tanner-agentic-checkpoints.sqlite")),
        enable_live_research=bool(_config_value("enable_live_research", True)),
        max_tool_response_chars=int(_config_value("max_tool_response_chars", 4_000)),
        max_command_output_chars=int(_config_value("max_command_output_chars", 4_000)),
        command_timeout=int(_config_value("command_timeout", 5)),
        max_concurrent_model_calls=int(_config_value("max_concurrent_model_calls", 4)),
        inter_call_delay_seconds=float(_config_value("inter_call_delay_seconds", 0.0)),
        max_rate_limit_retries=int(_config_value("max_rate_limit_retries", 2)),
        default_rate_limit_backoff_seconds=float(_config_value("default_rate_limit_backoff_seconds", 12.0)),
        max_length_limit_retries=int(_config_value("max_length_limit_retries", 2)),
        length_retry_token_increase=int(_config_value("length_retry_token_increase", 800)),
        max_length_retry_tokens=int(_config_value("max_length_retry_tokens", 6000)),
        roles=roles,
    )
