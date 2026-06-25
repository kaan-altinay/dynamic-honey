from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

_PROMPTS_PATH = Path(__file__).parent / "data" / "config.yaml"


class RolePromptSet(BaseModel):
    """System/user prompt templates for one generator role, split by pipeline version.

    ``*_v1`` is used when ``enable_scripted_flows`` is False (static-only bundles);
    ``*_v2`` is used when it is True (scripted-flow bundles). Templates still contain
    ``.format()``-style ``{placeholder}`` tokens for per-request substitution at the
    call site; this class only selects which static template applies.
    """

    model_config = ConfigDict(extra="forbid")

    system_v1: str
    system_v2: str
    user_v1: str
    user_v2: str

    def system(self, enable_scripted_flows: bool) -> str:
        return self.system_v2 if enable_scripted_flows else self.system_v1

    def user(self, enable_scripted_flows: bool) -> str:
        return self.user_v2 if enable_scripted_flows else self.user_v1


class PromptLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expert: RolePromptSet
    design: RolePromptSet
    coder: RolePromptSet
    review: RolePromptSet


def load_prompt_library(path: Path | None = None) -> PromptLibrary:
    config_path = path or _PROMPTS_PATH
    with open(config_path, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    prompts_raw = raw.get("prompts", raw)
    return PromptLibrary.model_validate(prompts_raw)
