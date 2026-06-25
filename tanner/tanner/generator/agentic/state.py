from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from tanner.generator.agentic.models import (
    ArtifactDraft,
    ExpertSpec,
    GeneratedBundle,
    GenerationRequest,
    PlannedArtifact,
    ReferencePack,
    ResourcePlan,
    ReviewDecision,
)


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
    generation_diagnostics: Annotated[list[dict], operator.add]
    plan_revision: int
    flow_descriptor: dict | None  # serialized FlowDescriptor when V2 active
