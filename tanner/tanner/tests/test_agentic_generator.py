import asyncio
import os
import tempfile
import unittest

from tanner.generator.agentic.models import (
    ArtifactDraft,
    GeneratedArtifact,
    GeneratedBundle,
    GeneratorRoleConfig,
    GeneratorRuntimeConfig,
    HeaderHint,
    HtmlPageContent,
    PlannedArtifact,
    ResourcePlan,
    StructuredHtmlPageDraft,
 )
from tanner.generator.agentic.fallback import build_fallback_bundle
from tanner.generator.agentic.renderers import render_artifact
from tanner.generator.agentic.validators import (
    ValidationError,
    ensure_generation_request,
    validate_bundle,
    validate_plan,
 )
from tanner.generator.agentic.workflow import AgenticBundleGenerator


class NoModelGenerator(AgenticBundleGenerator):
    def _get_role_model(self, role_name: str):
        return None


class AlwaysReviseGenerator(NoModelGenerator):
    def __init__(self, runtime_config):
        self.review_calls = 0
        super().__init__(runtime_config=runtime_config)

    async def _review_node(self, state):
        self.review_calls += 1
        return self._review_revise_or_fallback(state, ["forced revise"])


class TestAgenticBundleGenerator(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(None)
        self.temp_dir = tempfile.TemporaryDirectory(prefix="agentic-generator-")

    def tearDown(self):
        self.loop.close()
        self.temp_dir.cleanup()

    def _runtime_config(self, **overrides):
        checkpoint_path = os.path.join(self.temp_dir.name, "checkpoints.sqlite")
        role = GeneratorRoleConfig(provider="ollama", model="qwen2.5:14b-instruct", timeout=5, max_retries=0)
        values = {
            "backend": "agentic",
            "max_review_loops": 2,
            "max_bundle_artifacts": 4,
            "max_bundle_bytes": 262_144,
            "checkpoint_path": checkpoint_path,
            "enable_live_research": False,
            "max_tool_response_chars": 1024,
            "max_command_output_chars": 1024,
            "command_timeout": 2,
            "roles": {"expert": role, "design": role, "coder": role, "review": role},
        }
        values.update(overrides)
        return GeneratorRuntimeConfig.model_validate(values)

    def test_generate_bundle_for_wp_admin_includes_contextual_assets(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config())

        bundle = self.loop.run_until_complete(
            generator.generate_bundle(
                host="example.com",
                path="/wp-admin/login.php",
                site_profile={"index_page": "/index.html"},
            )
        )

        artifact_paths = {artifact.path for artifact in bundle.artifacts}
        self.assertEqual(bundle.primary_path, "/wp-admin/login.php")
        self.assertFalse(bundle.used_fallback)
        self.assertIn("/wp-admin/login.php", artifact_paths)
        self.assertIn("/wp-content/themes/twentytwenty/style.css", artifact_paths)
        self.assertIn("/wp-includes/js/wp-login.js", artifact_paths)
        self.assertIn("/wp-login.php", artifact_paths)

    def test_fallback_bundle_uses_profile_registry_for_config_theft(self):
        request = ensure_generation_request("example.com", "/.env", {"index_page": "/index.html"})

        bundle = build_fallback_bundle(request)

        artifact_paths = {artifact.path for artifact in bundle.artifacts}
        self.assertTrue(bundle.used_fallback)
        self.assertEqual(bundle.primary_path, "/.env")
        self.assertIn("/.env", artifact_paths)
        self.assertIn("/.env.bak", artifact_paths)
        self.assertIn("/storage/logs/app.log", artifact_paths)


    def test_fallback_bundle_respects_artifact_budget(self):
        request = ensure_generation_request("example.com", "/wp-admin/login.php", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=2)

        bundle = build_fallback_bundle(request, max_artifacts=runtime_config.max_bundle_artifacts)

        self.assertEqual(len(bundle.artifacts), 2)
        validate_bundle(bundle, request, runtime_config)

    def test_materialize_structured_draft_parses_closed_schema_payload(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config())
        structured_draft = StructuredHtmlPageDraft(
            artifact_id="wp-login-page",
            path="/wp-admin/login.php",
            content_model=HtmlPageContent(
                title="Login",
                heading="Login",
                paragraphs=["Prompt"],
                nav_links=[],
                linked_stylesheets=[],
                linked_scripts=[],
                footer="Footer",
            ),
            headers_hint=[HeaderHint(name="Content-Type", value="text/html; charset=utf-8")],
            review_notes=["note"],
        )

        draft = generator._materialize_structured_draft(structured_draft, plan_revision=3)

        self.assertEqual(draft.plan_revision, 3)
        self.assertEqual(draft.content_model["title"], "Login")
        self.assertEqual(draft.headers_hint, [{"Content-Type": "text/html; charset=utf-8"}])


    def test_content_model_skeleton_avoids_concrete_examples(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config())

        skeleton = generator._content_model_skeleton_for_kind("config_text")

        self.assertEqual(skeleton["format"], "<env-or-php-format>")
        self.assertEqual(skeleton["comment"], "<configuration comment>")
        self.assertEqual(skeleton["entries"][0]["key"], "<config-key>")
        self.assertEqual(skeleton["entries"][0]["value"], "<config-value>")


    def test_length_limit_error_detection_and_growth(self):
        generator = NoModelGenerator(
            runtime_config=self._runtime_config(
                length_retry_token_increase=900,
                max_length_retry_tokens=3200,
            )
        )

        class FakeLengthError(Exception):
            pass

        explicit_error = FakeLengthError("max completion tokens reached before generating a valid document")
        generic_error = Exception("ordinary failure")

        self.assertTrue(generator._is_length_limit_error(explicit_error))
        self.assertFalse(generator._is_length_limit_error(generic_error))
        self.assertEqual(generator._next_length_retry_tokens(1500), 2400)
        self.assertEqual(generator._next_length_retry_tokens(3000), 3200)
        self.assertIsNone(generator._next_length_retry_tokens(3200))

    def test_render_config_text_strips_comment_prefixes(self):
        artifact = render_artifact(
            ArtifactDraft(
                artifact_id="env",
                path="/.env",
                kind="config_text",
                content_model={
                    "format": "env",
                    "comment": "# Sample .env file",
                    "entries": [{"key": "APP_ENV", "value": "production"}],
                },
            )
        )

        self.assertEqual(
            artifact.body_bytes.decode("utf-8"),
            "# Sample .env file\nAPP_ENV=production\n",
        )


    def test_rate_limit_error_detection_and_retry_delay(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(default_rate_limit_backoff_seconds=7.0))

        class FakeRateLimitError(Exception):
            status_code = 429

        explicit_error = FakeRateLimitError(
            "Rate limit reached. Please try again in 11.655s."
        )
        generic_error = Exception("ordinary failure")

        self.assertTrue(generator._is_rate_limit_error(explicit_error))
        self.assertAlmostEqual(generator._rate_limit_sleep_seconds(explicit_error), 12.155, places=3)
        self.assertFalse(generator._is_rate_limit_error(generic_error))
        self.assertEqual(generator._rate_limit_sleep_seconds(generic_error), 7.0)


    def test_config_theft_plan_requires_contextual_support(self):
        request = ensure_generation_request("example.com", "/.env", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=5)
        plan = ResourcePlan(
            primary_path="/.env",
            theme_summary="Static bait .env file",
            artifacts=[
                PlannedArtifact(
                    artifact_id="env-primary",
                    path="/.env",
                    kind="config_text",
                    purpose="Primary config file",
                )
            ],
            bundle_budget_count=1,
            bundle_budget_bytes=1024,
            static_only=True,
            review_focus=["config_theft"],
        )

        with self.assertRaises(ValidationError):
            validate_plan(plan, request, runtime_config)

    def test_validate_bundle_rejects_internal_language_leak(self):
        request = ensure_generation_request("example.com", "/status", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=2)
        bundle = GeneratedBundle(
            primary_path="/status",
            artifacts=[
                GeneratedArtifact(
                    path="/status",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html; charset=utf-8"}],
                    body_bytes=b"<html><body><p>Fake admin status page</p></body></html>",
                    status_code=200,
                    source_artifact_id="status",
                    artifact_scope="static_file",
                )
            ],
            review_summary="pending",
            used_fallback=False,
        )

        with self.assertRaises(ValidationError):
            validate_bundle(bundle, request, runtime_config)


    def test_coder_fanout_generates_one_worker_per_planned_artifact(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config())
        request = ensure_generation_request("example.com", "/wp-admin/login.php", {"index_page": "/index.html"})
        expert_spec = self.loop.run_until_complete(generator._heuristic_expert_spec(request))
        resource_plan = generator._heuristic_plan(request, expert_spec)

        sends = generator._fan_out_coders(
            {
                "request": request,
                "expert_spec": expert_spec,
                "resource_plan": resource_plan,
                "plan_revision": 1,
            }
        )

        self.assertEqual(len(sends), len(resource_plan.artifacts))

    def test_review_loop_uses_fallback_after_max_review_loops(self):
        generator = AlwaysReviseGenerator(
            runtime_config=self._runtime_config(max_review_loops=2)
        )

        bundle = self.loop.run_until_complete(
            generator.generate_bundle(
                host="example.com",
                path="/admin/login",
                site_profile={"index_page": "/index.html"},
            )
        )

        self.assertTrue(bundle.used_fallback)
        self.assertEqual(bundle.primary_path, "/admin/login")
        self.assertEqual(generator.review_calls, 2)
