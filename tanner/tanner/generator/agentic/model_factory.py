from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model

from tanner.generator.agentic.models import GeneratorRuntimeConfig, RoleName


def build_role_model(
    role_name: RoleName,
    runtime_config: GeneratorRuntimeConfig,
    max_tokens_override: int | None = None,
 ):
    role_config = runtime_config.role_config(role_name)
    model_kwargs: dict[str, Any] = {
        "temperature": role_config.temperature,
        "timeout": role_config.timeout,
        "max_tokens": max_tokens_override if max_tokens_override is not None else role_config.max_tokens,
        "max_retries": role_config.max_retries,
    }
    return init_chat_model(
        role_config.model,
        model_provider=role_config.provider,
        **model_kwargs,
    )
