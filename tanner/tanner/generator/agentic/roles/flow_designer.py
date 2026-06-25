from __future__ import annotations

import re

from tanner.generator.agentic.models import (
    FlowCondition,
    FlowDescriptor,
    FlowResponse,
    FlowRule,
    GeneratedArtifact,
    GeneratedBundle,
    GenerationRequest,
)
from tanner.generator.agentic.validators import (
    _is_external_reference,
    normalize_path,
)
from tanner.generator.agentic.state import GraphState


class FlowDesignerRoleMixin:
    """Flow Designer (V2): synthesizes scripted-flow rules from the generated bundle."""

    _LOGIN_PATH_HINT_RE = re.compile(
        r"(?:^|/)(?:wp-login|login|logon|signin|sign-in|auth|account|session|formlogin|form-login)(?:$|[/?._-])",
        re.IGNORECASE,
    )

    async def _flow_designer_node(self, state: GraphState) -> dict:
        """
        Build a FlowDescriptor from explicit /_flow/ variant artifacts and
        synthesize minimal login POST flow behavior when no variants exist.
        """
        request = state["request"]
        resource_plan = state["resource_plan"]
        bundle = GeneratedBundle.model_validate(state["generated_bundle"])

        generated_paths = {a.path for a in bundle.artifacts}
        dynamic_paths = {
            a.path for a in resource_plan.artifacts if getattr(a, "dynamic_candidate", False)
        }

        login_post_targets: set[str] = set()
        for artifact in bundle.artifacts:
            login_post_targets.update(self._extract_post_form_targets(artifact, request))
        for target in login_post_targets:
            dynamic_paths.add(target)

        slug_to_path: dict[str, str] = {
            self._path_to_slug(p): p
            for p in generated_paths
            if not p.startswith("/_flow/")
        }

        rules: list[FlowRule] = []

        # First-class flow metadata from the ResourcePlan takes precedence over
        # variant-name conventions. This keeps non-login V2 flows declarative.
        for planned_artifact in resource_plan.artifacts:
            if not planned_artifact.path.startswith("/_flow/"):
                continue
            if planned_artifact.path not in generated_paths:
                continue
            if not planned_artifact.flow_match_path or planned_artifact.flow_response is None:
                continue

            response = planned_artifact.flow_response
            if response.artifact_path is None and response.redirect_to is None:
                response = response.model_copy(update={"artifact_path": planned_artifact.path})
            rules.append(
                FlowRule(
                    match_path=planned_artifact.flow_match_path,
                    condition=planned_artifact.flow_condition,
                    response=response,
                    priority=planned_artifact.flow_priority,
                )
            )

        # Process explicit /_flow/ variant artifacts first
        for artifact in bundle.artifacts:
            if not artifact.path.startswith("/_flow/"):
                continue
            if any(rule.response.artifact_path == artifact.path for rule in rules):
                continue
            parts = artifact.path.lstrip("/").split("/")
            # expected: ["_flow", slug, variant_name]
            if len(parts) != 3:
                self.logger.debug("flow_designer: skipping unexpected path %s", artifact.path)
                continue
            _, slug, variant_name = parts
            parent_path = slug_to_path.get(slug)
            if parent_path is None:
                self.logger.debug(
                    "flow_designer: no parent path for slug %r (path %s)", slug, artifact.path
                )
                continue
            rule = self._variant_to_rule(parent_path, variant_name, artifact.path, generated_paths)
            if rule is not None:
                rules.append(rule)

        # Synthesize generic login POST flow if missing.
        mutable_artifacts = list(bundle.artifacts)
        for path in sorted(login_post_targets):

            existing_post_rules = [
                r
                for r in rules
                if r.match_path == path
                and r.condition is not None
                and (r.condition.method or "").upper() == "POST"
            ]
            has_artifact_post_outcome = any(r.response.artifact_path for r in existing_post_rules)
            existing_outcomes = {
                (r.response.artifact_path, r.response.redirect_to, r.response.status_code)
                for r in existing_post_rules
            }

            slug = self._path_to_slug(path)
            fail_variant_path = "/_flow/{}/post-fail".format(slug)
            if not has_artifact_post_outcome and fail_variant_path not in generated_paths:
                source = next(
                    (a for a in mutable_artifacts if a.path == path and a.kind == "html_page"),
                    None,
                )
                if source is None:
                    source = self._find_html_form_source_for_target(mutable_artifacts, path, request)
                if source is not None:
                    synthetic = self._build_post_fail_variant_artifact(source, fail_variant_path)
                    mutable_artifacts.append(synthetic)
                    generated_paths.add(fail_variant_path)

            if not any(
                r.match_path == path
                and r.condition is not None
                and (r.condition.method or "").upper() == "POST"
                and r.response.artifact_path == fail_variant_path
                for r in rules
            ) and fail_variant_path in generated_paths:
                rules.append(
                    FlowRule(
                        match_path=path,
                        condition=FlowCondition(method="POST"),
                        response=FlowResponse(artifact_path=fail_variant_path, status_code=200),
                        priority=5,
                    )
                )
                existing_outcomes.add((fail_variant_path, None, 200))

            locked_variant_path = "/_flow/{}/post-locked".format(slug)
            has_locked_artifact = any(
                r.response.artifact_path == locked_variant_path for r in existing_post_rules
            )
            if not has_locked_artifact and locked_variant_path not in generated_paths:
                source = next(
                    (a for a in mutable_artifacts if a.path == path and a.kind == "html_page"),
                    None,
                )
                if source is None:
                    source = self._find_html_form_source_for_target(mutable_artifacts, path, request)
                if source is not None:
                    synthetic = self._build_post_locked_variant_artifact(source, locked_variant_path)
                    mutable_artifacts.append(synthetic)
                    generated_paths.add(locked_variant_path)

            if not any(
                r.match_path == path
                and r.condition is not None
                and (r.condition.method or "").upper() == "POST"
                and r.response.artifact_path == locked_variant_path
                for r in rules
            ) and locked_variant_path in generated_paths:
                rules.append(
                    FlowRule(
                        match_path=path,
                        condition=FlowCondition(
                            method="POST",
                            min_prior_post_count_to_path=3,
                            lockout_window_seconds=60,
                            lockout_active=True,
                        ),
                        response=FlowResponse(artifact_path=locked_variant_path, status_code=200),
                        priority=10,
                    )
                )

        # Implicit auth guard for dynamic_candidate admin/gated pages with no no-auth rule.
        # Do not attach the guard to POST form handlers themselves; otherwise a
        # missing-cookie redirect hides the invalid-credential POST variant.
        for path in sorted(dynamic_paths):
            if path in login_post_targets or self._is_login_path(path):
                continue
            if not self._is_gated_path(path):
                continue
            if any(
                r.match_path == path and r.condition and r.condition.missing_cookie is not None
                for r in rules
            ):
                continue
            login_path = next(
                (p for p in sorted(generated_paths) if self._is_login_path(p)), None
            )
            if login_path:
                rules.append(
                    FlowRule(
                        match_path=path,
                        condition=FlowCondition(missing_cookie="session_token"),
                        response=FlowResponse(redirect_to=login_path, status_code=302),
                        priority=10,
                    )
                )

        if not rules:
            self.logger.info("flow_designer: no rules produced for this bundle")
            return {
                "generated_bundle": bundle.model_copy(update={"flow_descriptor": None}),
                "flow_descriptor": None,
                "trace_notes": ["flow_designer:none"],
            }

        descriptor = FlowDescriptor(rules=rules)
        bundle = bundle.model_copy(update={"artifacts": mutable_artifacts, "flow_descriptor": descriptor})
        self.logger.info(
            "flow_designer: produced %d flow rule(s) for bundle %s",
            len(rules),
            bundle.primary_path,
        )
        return {
            "generated_bundle": bundle,
            "flow_descriptor": descriptor.model_dump(),
            "trace_notes": ["flow_designer:{}".format(len(rules))],
        }

    @staticmethod
    def _extract_post_form_targets(artifact: GeneratedArtifact, request: GenerationRequest) -> set[str]:
        if artifact.kind != "html_page" or artifact.path.startswith("/_flow/"):
            return set()

        body_text = artifact.body_bytes.decode("utf-8", errors="replace")
        form_tag_re = re.compile(r"<form\b[^>]*>", re.IGNORECASE)
        method_re = re.compile(r"\bmethod\s*=\s*[\"\']?([^\"\'\s>]+)", re.IGNORECASE)
        action_re = re.compile(r"\baction\s*=\s*[\"\']([^\"\']+)", re.IGNORECASE)

        targets: set[str] = set()
        for form_tag in form_tag_re.findall(body_text):
            method_match = method_re.search(form_tag)
            method = method_match.group(1).strip().upper() if method_match else "GET"
            if method != "POST":
                continue

            action_match = action_re.search(form_tag)
            action = action_match.group(1).strip() if action_match else artifact.path
            if not action or _is_external_reference(action):
                continue
            normalized = normalize_path(action, index_page=request.index_page)
            if normalized.startswith("/_flow/"):
                continue
            targets.add(normalized)
        return targets

    @staticmethod
    def _find_html_form_source_for_target(
        artifacts: list[GeneratedArtifact],
        target_path: str,
        request: GenerationRequest,
    ) -> GeneratedArtifact | None:
        form_tag_re = re.compile(r"<form\b[^>]*>", re.IGNORECASE)
        method_re = re.compile(r"\bmethod\s*=\s*[\"\']?([^\"\'\s>]+)", re.IGNORECASE)
        action_re = re.compile(r"\baction\s*=\s*[\"\']([^\"\']+)", re.IGNORECASE)

        for artifact in artifacts:
            if artifact.kind != "html_page" or artifact.path.startswith("/_flow/"):
                continue
            body_text = artifact.body_bytes.decode("utf-8", errors="replace")
            for form_tag in form_tag_re.findall(body_text):
                method_match = method_re.search(form_tag)
                method = method_match.group(1).strip().upper() if method_match else "GET"
                if method != "POST":
                    continue
                action_match = action_re.search(form_tag)
                action = action_match.group(1).strip() if action_match else artifact.path
                if not action or _is_external_reference(action):
                    continue
                normalized = normalize_path(action, index_page=request.index_page)
                if normalized == target_path:
                    return artifact
        return None

    @staticmethod
    def _build_post_locked_variant_artifact(source: GeneratedArtifact, variant_path: str) -> GeneratedArtifact:
        body_text = source.body_bytes.decode("utf-8", errors="replace")
        marker = (
            '<div class="message warning" role="alert">'
            'Too many invalid login attempts. Please wait 1 minute before trying again.'
            '</div>'
        )

        if "<form" in body_text:
            updated = body_text.replace("<form", marker + "\n<form", 1)
        elif "</body>" in body_text:
            updated = body_text.replace("</body>", marker + "\n</body>", 1)
        else:
            updated = body_text + "\n" + marker

        return GeneratedArtifact(
            path=variant_path,
            kind=source.kind,
            headers=list(source.headers),
            body_bytes=updated.encode("utf-8"),
            status_code=200,
            source_artifact_id="flow-synthesized:{}".format(source.source_artifact_id),
            artifact_scope=source.artifact_scope,
        )

    @staticmethod
    def _build_post_fail_variant_artifact(source: GeneratedArtifact, variant_path: str) -> GeneratedArtifact:
        body_text = source.body_bytes.decode("utf-8", errors="replace")
        marker = (
            '<div class="message error" role="alert">'
            'Authentication failed. Please verify your credentials and try again.'
            '</div>'
        )

        if "<form" in body_text:
            updated = body_text.replace("<form", marker + "\n<form", 1)
        elif "</body>" in body_text:
            updated = body_text.replace("</body>", marker + "\n</body>", 1)
        else:
            updated = body_text + "\n" + marker

        return GeneratedArtifact(
            path=variant_path,
            kind=source.kind,
            headers=list(source.headers),
            body_bytes=updated.encode("utf-8"),
            status_code=200,
            source_artifact_id="flow-synthesized:{}".format(source.source_artifact_id),
            artifact_scope=source.artifact_scope,
        )

    @staticmethod
    def _path_to_slug(path: str) -> str:
        """Derive a flow variant slug from a public path."""
        slug = path.lstrip("/")
        for ext in (".php", ".html", ".htm", ".asp", ".aspx", ".jsp", ".py"):
            if slug.endswith(ext):
                slug = slug[: -len(ext)]
                break
        slug = slug.replace("/", "-").rstrip("-") or "index"
        return slug

    def _variant_to_rule(
        self,
        parent_path: str,
        variant_name: str,
        artifact_path: str,
        all_paths: set,
    ) -> "FlowRule | None":
        """Map a /_flow/ variant name to the corresponding FlowRule."""
        if variant_name == "post-fail":
            return FlowRule(
                match_path=parent_path,
                condition=FlowCondition(method="POST"),
                response=FlowResponse(artifact_path=artifact_path, status_code=200),
                priority=5,
            )
        if variant_name == "post-locked":
            return FlowRule(
                match_path=parent_path,
                condition=FlowCondition(
                    method="POST",
                    min_prior_post_count_to_path=3,
                    lockout_window_seconds=60,
                    lockout_active=True,
                ),
                response=FlowResponse(artifact_path=artifact_path, status_code=200),
                priority=10,  # checked before post-fail
            )
        if variant_name == "no-auth":
            login_path = next((p for p in sorted(all_paths) if self._is_login_path(p)), "/")
            return FlowRule(
                match_path=parent_path,
                condition=FlowCondition(missing_cookie="session_token"),
                response=FlowResponse(redirect_to=login_path, status_code=302),
                priority=10,
            )
        if variant_name == "post-success":
            return FlowRule(
                match_path=parent_path,
                condition=FlowCondition(method="POST"),
                response=FlowResponse(
                    redirect_to=parent_path,
                    status_code=302,
                    set_cookie={"session_token": "granted"},
                ),
                priority=3,
            )
        if variant_name == "mfa-required":
            return FlowRule(
                match_path=parent_path,
                condition=FlowCondition(method="POST"),
                response=FlowResponse(
                    artifact_path=artifact_path,
                    status_code=200,
                    set_cookie={"session_stage": "mfa_required"},
                ),
                priority=4,
            )
        if variant_name == "logout":
            return FlowRule(
                match_path=parent_path,
                condition=None,
                response=FlowResponse(
                    redirect_to="/",
                    status_code=302,
                    clear_cookie=["session_token"],
                ),
                priority=5,
            )
        if variant_name == "session-expired":
            return FlowRule(
                match_path=parent_path,
                condition=FlowCondition(missing_cookie="session_token"),
                response=FlowResponse(artifact_path=artifact_path, status_code=200),
                priority=9,
            )
        return None

    @staticmethod
    def _is_login_path(path: str) -> bool:
        if path.startswith("/_flow/"):
            return False
        lowered = path.lower()
        if lowered.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".otf")):
            return False
        return bool(FlowDesignerRoleMixin._LOGIN_PATH_HINT_RE.search(lowered))

    @staticmethod
    def _is_form_handler_path(path: str) -> bool:
        if path.startswith("/_flow/"):
            return False
        lowered = path.lower()
        return any(
            hint in lowered
            for hint in (
                "formlogin",
                "form-login",
                "login.cgi",
                "loginform",
                "login_submit",
                "submitlogin",
                "checklogin",
                "auth.cgi",
            )
        )

    @staticmethod
    def _is_gated_path(path: str) -> bool:
        if path.startswith("/_flow/"):
            return False
        lower = path.lower()
        return any(kw in lower for kw in ("admin", "dashboard", "portal", "panel", "control", "manage"))
