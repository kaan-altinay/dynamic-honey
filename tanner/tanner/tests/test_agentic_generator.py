import asyncio
import base64
import json
import os
import tempfile
import unittest

from tanner.generator.agentic.models import (
    ArtifactDraft,
    GeneratedArtifact,
    GeneratedBundle,
    FlowCondition,
    FlowDescriptor,
    FlowResponse,
    FlowRule,
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
    diagnose_flow_reachability,
    validate_bundle,
    validate_artifact_draft,
    validate_plan,
 )
from tanner.generator.agentic.workflow import AgenticBundleGenerator
from tanner.flow.flow_evaluator import FlowEvaluator


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
            "review_log_path": os.path.join(self.temp_dir.name, "review-log.json"),
            "enable_live_research": False,
            "max_tool_response_chars": 1024,
            "max_command_output_chars": 1024,
            "command_timeout": 2,
            "roles": {"expert": role, "design": role, "coder": role, "review": role},
        }
        values.update(overrides)
        return GeneratorRuntimeConfig.model_validate(values)


    def test_v2_intent_routing_is_gated_by_scripted_flows(self):
        v1_generator = NoModelGenerator(runtime_config=self._runtime_config(enable_scripted_flows=False))
        v2_generator = NoModelGenerator(runtime_config=self._runtime_config(enable_scripted_flows=True))

        v1_container = self.loop.run_until_complete(
            v1_generator._heuristic_expert_spec(
                ensure_generation_request("example.com", "/containers/json", {"index_page": "/index.html"})
            )
        )
        v2_container = self.loop.run_until_complete(
            v2_generator._heuristic_expert_spec(
                ensure_generation_request("example.com", "/containers/json", {"index_page": "/index.html"})
            )
        )
        v1_secret = self.loop.run_until_complete(
            v1_generator._heuristic_expert_spec(
                ensure_generation_request("example.com", "/.env", {"index_page": "/index.html"})
            )
        )
        v2_secret = self.loop.run_until_complete(
            v2_generator._heuristic_expert_spec(
                ensure_generation_request("example.com", "/.env", {"index_page": "/index.html"})
            )
        )

        self.assertEqual(v1_container.intent_family, "generic_recon")
        self.assertEqual(v2_container.intent_family, "container_api")
        self.assertEqual(v1_secret.intent_family, "config_theft")
        self.assertEqual(v2_secret.intent_family, "config_secret")

    def test_v2_intent_routing_covers_endpoint_families(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(enable_scripted_flows=True))
        cases = {
            "/wp-login.php": "auth_portal",
            "/boaform/admin/formLogin": "network_device",
            "/api/v1/pods": "kubernetes_api",
            "/_nodes": "elastic_api",
            "/solr/admin/cores": "solr_api",
            "/shell": "webshell_probe",
        }

        for path, expected_intent in cases.items():
            with self.subTest(path=path):
                spec = self.loop.run_until_complete(
                    generator._heuristic_expert_spec(
                        ensure_generation_request("example.com", path, {"index_page": "/index.html"})
                    )
                )
                self.assertEqual(spec.intent_family, expected_intent)
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
        planned_artifact = PlannedArtifact(
            artifact_id="wp-login-page",
            path="/wp-admin/login.php",
            kind="html_page",
            purpose="Primary WordPress login page",
        )

        draft = generator._materialize_structured_draft(
            structured_draft,
            artifact=planned_artifact,
            plan_revision=3,
        )

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

    def test_cms_probe_login_plan_requires_stylesheet_artifact(self):
        request = ensure_generation_request("example.com", "/wp-admin/login.php", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=6)
        plan = ResourcePlan(
            primary_path="/wp-admin/login.php",
            theme_summary="WordPress login surface",
            artifacts=[
                PlannedArtifact(
                    artifact_id="login-admin",
                    path="/wp-admin/login.php",
                    kind="html_page",
                    purpose="Primary login page",
                    links_to=["/wp-login.php"],
                ),
                PlannedArtifact(
                    artifact_id="login-canonical",
                    path="/wp-login.php",
                    kind="html_page",
                    purpose="Canonical login page",
                    links_to=["/wp-admin/login.php"],
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=8_192,
            static_only=True,
            review_focus=["cms_probe"],
        )

        with self.assertRaises(ValidationError):
            validate_plan(plan, request, runtime_config)

    def test_cms_probe_login_plan_accepts_linked_stylesheet_artifact(self):
        request = ensure_generation_request("example.com", "/wp-admin/login.php", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=6)
        stylesheet_path = "/wp-content/themes/twentytwenty/style.css"
        plan = ResourcePlan(
            primary_path="/wp-admin/login.php",
            theme_summary="WordPress login surface",
            artifacts=[
                PlannedArtifact(
                    artifact_id="login-admin",
                    path="/wp-admin/login.php",
                    kind="html_page",
                    purpose="Primary login page",
                    links_to=["/wp-login.php", stylesheet_path],
                ),
                PlannedArtifact(
                    artifact_id="login-canonical",
                    path="/wp-login.php",
                    kind="html_page",
                    purpose="Canonical login page",
                    links_to=["/wp-admin/login.php", stylesheet_path],
                ),
                PlannedArtifact(
                    artifact_id="wp-style",
                    path=stylesheet_path,
                    kind="stylesheet",
                    purpose="Shared login stylesheet",
                ),
            ],
            bundle_budget_count=3,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["cms_probe"],
        )

        validate_plan(plan, request, runtime_config)


    def test_plan_rejects_generated_asset_file_artifacts(self):
        request = ensure_generation_request("example.com", "/wp-admin/login.php", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=6)
        stylesheet_path = "/wp-content/themes/twentytwenty/style.css"
        plan = ResourcePlan(
            primary_path="/wp-admin/login.php",
            theme_summary="WordPress login surface",
            artifacts=[
                PlannedArtifact(
                    artifact_id="login-admin",
                    path="/wp-admin/login.php",
                    kind="html_page",
                    purpose="Primary login page",
                    links_to=[stylesheet_path],
                ),
                PlannedArtifact(
                    artifact_id="wp-style",
                    path=stylesheet_path,
                    kind="stylesheet",
                    purpose="Shared login stylesheet",
                ),
                PlannedArtifact(
                    artifact_id="logo",
                    path="/wp-admin/images/wordpress-logo.svg",
                    kind="asset_file",
                    purpose="Logo binary asset",
                ),
            ],
            bundle_budget_count=3,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["cms_probe"],
        )

        with self.assertRaises(ValidationError):
            validate_plan(plan, request, runtime_config)


    def test_xml_plan_requires_xml_document_primary_kind(self):
        request = ensure_generation_request("example.com", "/gatedesc.xml", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=6)
        plan = ResourcePlan(
            primary_path="/gatedesc.xml",
            theme_summary="UPnP descriptor surface",
            artifacts=[
                PlannedArtifact(
                    artifact_id="primary-xml",
                    path="/gatedesc.xml",
                    kind="config_text",
                    purpose="Incorrectly modeled XML payload as config text",
                ),
                PlannedArtifact(
                    artifact_id="support-xml",
                    path="/WANCfgSCPD.xml",
                    kind="xml_document",
                    purpose="Supporting descriptor",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["framework_probe"],
        )

        with self.assertRaises(ValidationError):
            validate_plan(plan, request, runtime_config)

    def test_xml_plan_accepts_xml_document_primary_kind(self):
        request = ensure_generation_request("example.com", "/gatedesc.xml", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=6)
        plan = ResourcePlan(
            primary_path="/gatedesc.xml",
            theme_summary="UPnP descriptor surface",
            artifacts=[
                PlannedArtifact(
                    artifact_id="primary-xml",
                    path="/gatedesc.xml",
                    kind="xml_document",
                    purpose="Primary gateway descriptor",
                    links_to=["/WANCfgSCPD.xml"],
                ),
                PlannedArtifact(
                    artifact_id="support-xml",
                    path="/WANCfgSCPD.xml",
                    kind="xml_document",
                    purpose="Supporting descriptor",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["framework_probe"],
        )

        validate_plan(plan, request, runtime_config)

    def test_render_xml_document_artifact_outputs_application_xml(self):
        artifact = render_artifact(
            ArtifactDraft(
                artifact_id="xml-primary",
                path="/gatedesc.xml",
                kind="xml_document",
                content_model={
                    "lines": [
                        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                        "<root><child>value</child></root>",
                    ]
                },
            )
        )

        self.assertEqual(artifact.kind, "xml_document")
        self.assertEqual(
            artifact.body_bytes.decode("utf-8"),
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root><child>value</child></root>\n",
        )
        self.assertTrue(
            any(
                isinstance(header, dict) and header.get("Content-Type") == "application/xml; charset=utf-8"
                for header in artifact.headers
            )
        )

    def test_fallback_bundle_for_xml_request_uses_xml_primary(self):
        request = ensure_generation_request("example.com", "/gatedesc.xml", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=4)
        bundle = build_fallback_bundle(request, max_artifacts=runtime_config.max_bundle_artifacts)

        primary = next((artifact for artifact in bundle.artifacts if artifact.path == "/gatedesc.xml"), None)
        self.assertIsNotNone(primary)
        self.assertEqual(primary.kind, "xml_document")
        validate_bundle(bundle, request, runtime_config)


    def test_json_plan_requires_json_document_primary_kind(self):
        request = ensure_generation_request("example.com", "/api/status.json", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=6)
        plan = ResourcePlan(
            primary_path="/api/status.json",
            theme_summary="JSON endpoint surface",
            artifacts=[
                PlannedArtifact(
                    artifact_id="primary-json",
                    path="/api/status.json",
                    kind="html_page",
                    purpose="Incorrectly modeled JSON endpoint",
                ),
                PlannedArtifact(
                    artifact_id="support-text",
                    path="/version",
                    kind="plain_text",
                    purpose="Support metadata endpoint",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["framework_probe"],
        )

        with self.assertRaises(ValidationError):
            validate_plan(plan, request, runtime_config)

    def test_plain_text_plan_requires_plain_text_primary_kind(self):
        request = ensure_generation_request("example.com", "/status.txt", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=6)
        plan = ResourcePlan(
            primary_path="/status.txt",
            theme_summary="Plain text endpoint surface",
            artifacts=[
                PlannedArtifact(
                    artifact_id="primary-text",
                    path="/status.txt",
                    kind="html_page",
                    purpose="Incorrectly modeled text endpoint",
                ),
                PlannedArtifact(
                    artifact_id="support-text",
                    path="/version",
                    kind="plain_text",
                    purpose="Support metadata endpoint",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["generic_recon"],
        )

        with self.assertRaises(ValidationError):
            validate_plan(plan, request, runtime_config)

    def test_render_json_document_artifact_outputs_application_json(self):
        artifact = render_artifact(
            ArtifactDraft(
                artifact_id="json-primary",
                path="/api/status.json",
                kind="json_document",
                content_model={"document": {"status": "ok", "version": "1.0.0"}},
            )
        )

        self.assertEqual(artifact.kind, "json_document")
        self.assertEqual(json.loads(artifact.body_bytes.decode("utf-8")), {"status": "ok", "version": "1.0.0"})
        self.assertTrue(
            any(
                isinstance(header, dict) and header.get("Content-Type") == "application/json; charset=utf-8"
                for header in artifact.headers
            )
        )

    def test_render_binary_asset_artifact_outputs_binary_bytes(self):
        payload = b"\x00\x01\x02abc"
        artifact = render_artifact(
            ArtifactDraft(
                artifact_id="binary-primary",
                path="/favicon.ico",
                kind="binary_asset",
                content_model={
                    "content_type": "image/x-icon",
                    "content_base64": base64.b64encode(payload).decode("ascii"),
                },
            )
        )

        self.assertEqual(artifact.kind, "binary_asset")
        self.assertEqual(artifact.body_bytes, payload)
        self.assertTrue(
            any(
                isinstance(header, dict) and header.get("Content-Type") == "image/x-icon"
                for header in artifact.headers
            )
        )

    def test_fallback_bundle_for_json_request_uses_json_primary(self):
        request = ensure_generation_request("example.com", "/api/status.json", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=4)
        bundle = build_fallback_bundle(request, max_artifacts=runtime_config.max_bundle_artifacts)

        primary = next((artifact for artifact in bundle.artifacts if artifact.path == "/api/status.json"), None)
        self.assertIsNotNone(primary)
        self.assertEqual(primary.kind, "json_document")
        validate_bundle(bundle, request, runtime_config)

    def test_fallback_bundle_for_binary_request_uses_binary_primary(self):
        request = ensure_generation_request("example.com", "/favicon.ico", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=4)
        bundle = build_fallback_bundle(request, max_artifacts=runtime_config.max_bundle_artifacts)

        primary = next((artifact for artifact in bundle.artifacts if artifact.path == "/favicon.ico"), None)
        self.assertIsNotNone(primary)
        self.assertEqual(primary.kind, "binary_asset")
        validate_bundle(bundle, request, runtime_config)


    def test_validate_plan_rejects_response_contract_content_type_mismatch(self):
        request = ensure_generation_request("example.com", "/api/status.json", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=4)
        plan = ResourcePlan(
            primary_path="/api/status.json",
            theme_summary="JSON endpoint surface",
            artifacts=[
                PlannedArtifact(
                    artifact_id="primary-json",
                    path="/api/status.json",
                    kind="json_document",
                    purpose="Primary JSON endpoint",
                    response_contract={"status_code": 200, "content_type": "text/html; charset=utf-8"},
                ),
                PlannedArtifact(
                    artifact_id="support-text",
                    path="/version",
                    kind="plain_text",
                    purpose="Support metadata endpoint",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["framework_probe"],
        )

        with self.assertRaises(ValidationError):
            validate_plan(plan, request, runtime_config)

    def test_render_artifact_uses_draft_status_code_and_content_type(self):
        artifact = render_artifact(
            ArtifactDraft(
                artifact_id="text-status",
                path="/status.txt",
                kind="plain_text",
                content_model={"lines": ["status: ok"]},
                status_code=401,
                content_type="text/plain; charset=utf-8",
                headers_hint=[{"WWW-Authenticate": "Basic realm=\"Restricted\""}],
            )
        )

        self.assertEqual(artifact.status_code, 401)
        self.assertTrue(
            any(
                isinstance(header, dict)
                and header.get("Content-Type") == "text/plain; charset=utf-8"
                for header in artifact.headers
            )
        )
        self.assertTrue(
            any(
                isinstance(header, dict)
                and header.get("WWW-Authenticate") == "Basic realm=\"Restricted\""
                for header in artifact.headers
            )
        )

    def test_validate_bundle_rejects_generated_content_type_mismatch(self):
        request = ensure_generation_request("example.com", "/api/status.json", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=2)
        bundle = GeneratedBundle(
            primary_path="/api/status.json",
            artifacts=[
                GeneratedArtifact(
                    path="/api/status.json",
                    kind="json_document",
                    headers=[{"Content-Type": "text/plain; charset=utf-8"}],
                    body_bytes=b'{"status":"ok"}',
                    status_code=200,
                    source_artifact_id="json-status",
                    artifact_scope="static_file",
                )
            ],
            review_summary="pending",
            used_fallback=False,
        )

        with self.assertRaises(ValidationError):
            validate_bundle(bundle, request, runtime_config)
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

    def test_validate_bundle_allows_artifact_count_overflow(self):
        request = ensure_generation_request("example.com", "/admin", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=1)
        bundle = GeneratedBundle(
            primary_path="/admin",
            artifacts=[
                GeneratedArtifact(
                    path="/admin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html; charset=utf-8"}],
                    body_bytes=b"<html><body><p>Admin</p></body></html>",
                    status_code=200,
                    source_artifact_id="admin",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/robots.txt",
                    kind="robots_txt",
                    headers=[{"Content-Type": "text/plain; charset=utf-8"}],
                    body_bytes=b"User-agent: *\nDisallow:\n",
                    status_code=200,
                    source_artifact_id="robots",
                    artifact_scope="static_file",
                ),
            ],
            review_summary="pending",
            used_fallback=False,
        )

        validate_bundle(bundle, request, runtime_config)



    def test_validate_bundle_allows_index_page_baseline_link(self):
        request = ensure_generation_request("example.com", "/wp-admin/login.php", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=2)
        bundle = GeneratedBundle(
            primary_path="/wp-admin/login.php",
            artifacts=[
                GeneratedArtifact(
                    path="/wp-admin/login.php",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html; charset=utf-8"}],
                    body_bytes=b'<html><body><a href="/index.html">Home</a></body></html>',
                    status_code=200,
                    source_artifact_id="page",
                    artifact_scope="static_file",
                )
            ],
            review_summary="pending",
            used_fallback=False,
        )

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

    def test_review_loop_retains_last_bundle_after_max_review_loops(self):
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

        self.assertFalse(bundle.used_fallback)
        self.assertEqual(bundle.primary_path, "/admin/login")
        self.assertEqual(generator.review_calls, 2)
        self.assertIn("review loop budget exhausted", bundle.review_summary)


    def test_review_loop_hard_failures_fall_back_after_max_review_loops(self):
        generator = NoModelGenerator(
            runtime_config=self._runtime_config(max_review_loops=1)
        )
        request = ensure_generation_request("example.com", "/admin/login", {"index_page": "/index.html"})

        result = generator._review_revise_or_fallback(
            {"request": request, "review_iteration": 0},
            ["bundle exceeds max_bundle_artifacts"],
            hard_failure=True,
        )

        decision = result["review_decision"]
        self.assertEqual(decision.decision, "fallback")
        self.assertIn("structural validation failed", decision.reasons[0])

    def test_review_log_appends_history_per_endpoint(self):
        runtime_config = self._runtime_config(max_review_loops=1)
        generator = AlwaysReviseGenerator(runtime_config=runtime_config)

        self.loop.run_until_complete(
            generator.generate_bundle(
                host="example.com",
                path="/admin/login",
                site_profile={"index_page": "/index.html"},
            )
        )
        self.loop.run_until_complete(
            generator.generate_bundle(
                host="example.com",
                path="/admin/login",
                site_profile={"index_page": "/index.html"},
            )
        )

        review_log_path = runtime_config.review_log_path
        with open(review_log_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertIn("/admin/login", payload)
        history = payload["/admin/login"]
        self.assertGreaterEqual(len(history), 2)
        for entry in history:
            self.assertIn("date", entry)
            self.assertIn("review_output", entry)
            self.assertIn("decision", entry)

    def test_php_credential_bait_allows_text_plain_contract(self):
        request = ensure_generation_request("example.com", "/admin/config.php", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=3)
        plan = ResourcePlan(
            primary_path="/admin/config.php",
            theme_summary="Exposed PHP configuration source",
            artifacts=[
                PlannedArtifact(
                    artifact_id="admin-config",
                    path="/admin/config.php",
                    kind="credential_bait",
                    purpose="PHP configuration source leak",
                    response_contract={"status_code": 200, "content_type": "text/plain; charset=utf-8"},
                ),
                PlannedArtifact(
                    artifact_id="admin-log",
                    path="/admin/error.log",
                    kind="log_excerpt",
                    purpose="Supporting PHP error log",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["credential_bait"],
        )

        validate_plan(plan, request, runtime_config)


    def test_server_side_script_extension_does_not_force_html_content_type(self):
        """
        .php / .asp / .jsp extensions must not impose text/html on the plan.
        Their actual content type is dynamic; only the artifact kind should
        determine the expectation.  Regression for: validate_plan rejecting
        /config/database.php with text/plain contract because .php was mapped
        to text/html in the extension branch.
        """
        from tanner.generator.agentic.validators import _test_expected_content_type_for_kind_and_path_cases
        _test_expected_content_type_for_kind_and_path_cases()

        # Plan-level check: config_text at a .php path with text/plain contract must pass.
        request = ensure_generation_request("example.com", "/config/database.php", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=3)
        plan = ResourcePlan(
            primary_path="/config/database.php",
            theme_summary="Exposed PHP database config",
            artifacts=[
                PlannedArtifact(
                    artifact_id="db-config",
                    path="/config/database.php",
                    kind="config_text",
                    purpose="PHP database configuration file",
                    response_contract={"status_code": 200, "content_type": "text/plain; charset=utf-8"},
                ),
                PlannedArtifact(
                    artifact_id="db-log",
                    path="/config/db.log",
                    kind="log_excerpt",
                    purpose="Supporting database log",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["config_theft"],
        )
        validate_plan(plan, request, runtime_config)

        # html_page at a .php path with text/html contract must also pass.
        plan2 = ResourcePlan(
            primary_path="/config/database.php",
            theme_summary="PHP login page",
            artifacts=[
                PlannedArtifact(
                    artifact_id="login",
                    path="/config/database.php",
                    kind="html_page",
                    purpose="PHP login page",
                    response_contract={"status_code": 200, "content_type": "text/html; charset=utf-8"},
                ),
                PlannedArtifact(
                    artifact_id="style",
                    path="/assets/style.css",
                    kind="stylesheet",
                    purpose="stylesheet",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["auth_portal"],
        )
        validate_plan(plan2, request, runtime_config)
    def test_vendor_json_content_type_matches_json_contracts(self):
        request = ensure_generation_request("example.com", "/actuator/gateway/routes", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=2)
        plan = ResourcePlan(
            primary_path="/actuator/gateway/routes",
            theme_summary="Spring Boot actuator API",
            artifacts=[
                PlannedArtifact(
                    artifact_id="routes",
                    path="/actuator/gateway/routes",
                    kind="json_document",
                    purpose="Gateway routes actuator payload",
                    response_contract={
                        "status_code": 200,
                        "content_type": "application/vnd.spring-boot.actuator.v3+json",
                    },
                ),
                PlannedArtifact(
                    artifact_id="health",
                    path="/actuator/health",
                    kind="json_document",
                    purpose="Health actuator payload",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["framework_probe"],
        )
        bundle = GeneratedBundle(
            primary_path="/actuator/gateway/routes",
            artifacts=[
                GeneratedArtifact(
                    path="/actuator/gateway/routes",
                    kind="json_document",
                    headers=[{"Content-Type": "application/vnd.spring-boot.actuator.v3+json"}],
                    body_bytes=b'{"routes":[]}',
                    status_code=200,
                    source_artifact_id="routes",
                    artifact_scope="static_file",
                )
            ],
            review_summary="pending",
            used_fallback=False,
        )

        validate_plan(plan, request, runtime_config)
        validate_bundle(bundle, request, runtime_config)

    def test_xml_structured_content_types_match_xml_contracts(self):
        request = ensure_generation_request("example.com", "/wsman", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=2)
        plan = ResourcePlan(
            primary_path="/wsman",
            theme_summary="WS-Man endpoint",
            artifacts=[
                PlannedArtifact(
                    artifact_id="wsdl",
                    path="/wsman/service.wsdl",
                    kind="xml_document",
                    purpose="WSDL service description",
                    response_contract={
                        "status_code": 200,
                        "content_type": "application/wsdl+xml; charset=utf-8",
                    },
                ),
                PlannedArtifact(
                    artifact_id="primary",
                    path="/wsman",
                    kind="xml_document",
                    purpose="Primary WS-Man SOAP endpoint",
                    response_contract={
                        "status_code": 200,
                        "content_type": "application/soap+xml; charset=utf-8",
                    },
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["device_protocol"],
        )
        bundle = GeneratedBundle(
            primary_path="/wsman",
            artifacts=[
                GeneratedArtifact(
                    path="/wsman",
                    kind="xml_document",
                    headers=[{"Content-Type": "application/soap+xml; charset=utf-8"}],
                    body_bytes=b"<?xml version=\"1.0\"?><s:Envelope></s:Envelope>",
                    status_code=200,
                    source_artifact_id="primary",
                    artifact_scope="static_file",
                )
            ],
            review_summary="pending",
            used_fallback=False,
        )

        validate_plan(plan, request, runtime_config)
        validate_bundle(bundle, request, runtime_config)

    def test_text_family_content_types_match_text_contracts(self):
        request = ensure_generation_request("example.com", "/boaform/admin/formLogin", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(max_bundle_artifacts=2)
        plan = ResourcePlan(
            primary_path="/boaform/admin/formLogin",
            theme_summary="Boa login handler",
            artifacts=[
                PlannedArtifact(
                    artifact_id="handler",
                    path="/boaform/admin/formLogin",
                    kind="plain_text",
                    purpose="Login handler response",
                    response_contract={
                        "status_code": 200,
                        "content_type": "text/html; charset=UTF-8",
                    },
                ),
                PlannedArtifact(
                    artifact_id="robots",
                    path="/robots.txt",
                    kind="robots_txt",
                    purpose="Robots policy",
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16_384,
            static_only=True,
            review_focus=["router_login"],
        )

        validate_plan(plan, request, runtime_config)

    def test_text_extension_allows_semantic_text_kinds(self):
        request = ensure_generation_request("example.com", "/.env", {"index_page": "/index.html"})
        draft = ArtifactDraft(
            artifact_id="backup-manifest",
            path="/backups/manifest.txt",
            kind="backup_manifest",
            content_model={"lines": ["2026-05-25T00:00:00Z app.tar.gz"]},
            content_type="text/plain; charset=utf-8",
        )

        validate_artifact_draft(draft, request)

    def test_design_validation_loop_budget_exhaustion_falls_through_with_heuristic_plan(self):
        request = ensure_generation_request("example.com", "/wp-login.php", {"index_page": "/index.html"})
        generator = NoModelGenerator(
            runtime_config=self._runtime_config(max_review_loops=1, max_design_validation_loops=2)
        )
        expert_spec = self.loop.run_until_complete(generator._heuristic_expert_spec(request))

        state_0 = {"request": request, "expert_spec": expert_spec}
        first = generator._design_revise_or_fallback(state_0, ["invalid dynamic scope"])
        self.assertEqual(first["design_validation_decision"], "revise")

        state_1 = {
            "request": request,
            "expert_spec": expert_spec,
            "design_validation_iteration": first["design_validation_iteration"],
        }
        second = generator._design_revise_or_fallback(state_1, ["still invalid"])
        # Budget exhausted: must route to approve (proceed with heuristic plan)
        # rather than triggering a deterministic stub fallback.
        self.assertEqual(second["design_validation_decision"], "approve")
        self.assertIn("resource_plan", second)
        self.assertIsNotNone(second["resource_plan"])
        self.assertIn("heuristic_fallthrough", " ".join(second.get("trace_notes", [])))

    def test_runtime_config_disallows_fallback_persistence_by_default(self):
        runtime_config = self._runtime_config()

        self.assertFalse(runtime_config.allow_fallback_persistence)


    def test_form_handler_flow_rejects_redirect_only_without_post_failure_artifact(self):
        request = ensure_generation_request(
            "example.com",
            "/boaform/admin/formLogin",
            {"index_page": "/index.html"},
        )
        runtime_config = self._runtime_config(enable_scripted_flows=True, max_bundle_artifacts=4)
        bundle = GeneratedBundle(
            primary_path="/boaform/admin/formLogin",
            artifacts=[
                GeneratedArtifact(
                    path="/boaform/admin/formLogin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=(
                        b'<html><body><form method="post" action="/boaform/admin/formLogin">'
                        b'<input name="username"><input name="password" type="password">'
                        b'</form></body></html>'
                    ),
                    status_code=200,
                    source_artifact_id="handler",
                    artifact_scope="dynamic_endpoint",
                ),
                GeneratedArtifact(
                    path="/admin/login.asp",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><body>Login</body></html>',
                    status_code=200,
                    source_artifact_id="login",
                    artifact_scope="static_file",
                ),
            ],
            review_summary="pending",
            used_fallback=False,
            flow_descriptor=FlowDescriptor(
                rules=[
                    FlowRule(
                        match_path="/boaform/admin/formLogin",
                        condition=FlowCondition(missing_cookie="session_token"),
                        response=FlowResponse(redirect_to="/admin/login.asp", status_code=302),
                        priority=10,
                    )
                ]
            ),
        )

        with self.assertRaisesRegex(ValidationError, "requires at least one POST flow rule"):
            validate_bundle(bundle, request, runtime_config)

    def test_form_handler_flow_requires_visible_failure_feedback_artifact(self):
        request = ensure_generation_request(
            "example.com",
            "/boaform/admin/formLogin",
            {"index_page": "/index.html"},
        )
        runtime_config = self._runtime_config(enable_scripted_flows=True, max_bundle_artifacts=4)
        bundle = GeneratedBundle(
            primary_path="/boaform/admin/formLogin",
            artifacts=[
                GeneratedArtifact(
                    path="/boaform/admin/formLogin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=(
                        b'<html><body><form method="post" action="/boaform/admin/formLogin">'
                        b'<input name="username"><input name="password" type="password">'
                        b'</form></body></html>'
                    ),
                    status_code=200,
                    source_artifact_id="handler",
                    artifact_scope="dynamic_endpoint",
                ),
                GeneratedArtifact(
                    path="/_flow/boaform-admin-formLogin/post-fail",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><body>User name or password is error. Please try again.</body></html>',
                    status_code=200,
                    source_artifact_id="post-fail",
                    artifact_scope="dynamic_endpoint",
                ),
                GeneratedArtifact(
                    path="/_flow/boaform-admin-formLogin/post-locked",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><body>Still no luck.</body></html>',
                    status_code=200,
                    source_artifact_id="post-locked",
                    artifact_scope="dynamic_endpoint",
                ),
            ],
            review_summary="pending",
            used_fallback=False,
            flow_descriptor=FlowDescriptor(
                rules=[
                    FlowRule(
                        match_path="/boaform/admin/formLogin",
                        condition=FlowCondition(method="POST"),
                        response=FlowResponse(
                            artifact_path="/_flow/boaform-admin-formLogin/post-fail",
                            status_code=200,
                        ),
                        priority=5,
                    ),
                    FlowRule(
                        match_path="/boaform/admin/formLogin",
                        condition=FlowCondition(
                            method="POST",
                            min_prior_post_count_to_path=3,
                            lockout_window_seconds=60,
                            lockout_active=True,
                        ),
                        response=FlowResponse(artifact_path="/_flow/boaform-admin-formLogin/post-locked", status_code=200),
                        priority=10,
                    ),
                ]
            ),
        )

        with self.assertRaisesRegex(ValidationError, "visible too-many-attempts lockout artifact"):
            validate_bundle(bundle, request, runtime_config)

    def test_form_handler_flow_accepts_post_failure_artifact_and_distinct_outcome(self):
        request = ensure_generation_request(
            "example.com",
            "/boaform/admin/formLogin",
            {"index_page": "/index.html"},
        )
        runtime_config = self._runtime_config(enable_scripted_flows=True, max_bundle_artifacts=4)
        bundle = GeneratedBundle(
            primary_path="/boaform/admin/formLogin",
            artifacts=[
                GeneratedArtifact(
                    path="/boaform/admin/formLogin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=(
                        b'<html><body><form method="post" action="/boaform/admin/formLogin">'
                        b'<input name="username"><input name="password" type="password">'
                        b'</form></body></html>'
                    ),
                    status_code=200,
                    source_artifact_id="handler",
                    artifact_scope="dynamic_endpoint",
                ),
                GeneratedArtifact(
                    path="/_flow/boaform-admin-formLogin/post-fail",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><body>User name or password is error. Please try again.</body></html>',
                    status_code=200,
                    source_artifact_id="post-fail",
                    artifact_scope="dynamic_endpoint",
                ),
                GeneratedArtifact(
                    path="/_flow/boaform-admin-formLogin/post-locked",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><body>Too many invalid login attempts. Please wait 1 minute before trying again.</body></html>',
                    status_code=200,
                    source_artifact_id="post-locked",
                    artifact_scope="dynamic_endpoint",
                ),
            ],
            review_summary="pending",
            used_fallback=False,
            flow_descriptor=FlowDescriptor(
                rules=[
                    FlowRule(
                        match_path="/boaform/admin/formLogin",
                        condition=FlowCondition(method="POST"),
                        response=FlowResponse(
                            artifact_path="/_flow/boaform-admin-formLogin/post-fail",
                            status_code=200,
                        ),
                        priority=5,
                    ),
                    FlowRule(
                        match_path="/boaform/admin/formLogin",
                        condition=FlowCondition(
                            method="POST",
                            min_prior_post_count_to_path=3,
                            lockout_window_seconds=60,
                            lockout_active=True,
                        ),
                        response=FlowResponse(artifact_path="/_flow/boaform-admin-formLogin/post-locked", status_code=200),
                        priority=10,
                    ),
                ]
            ),
        )

        validate_bundle(bundle, request, runtime_config)

    def test_flow_reachability_diagnostics_are_nonblocking(self):
        request = ensure_generation_request("example.com", "/admin", {"index_page": "/index.html"})
        runtime_config = self._runtime_config(enable_scripted_flows=True, max_bundle_artifacts=3)
        bundle = GeneratedBundle(
            primary_path="/admin",
            artifacts=[
                GeneratedArtifact(
                    path="/admin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b"<html><body>Admin</body></html>",
                    status_code=200,
                    source_artifact_id="admin",
                    artifact_scope="dynamic_endpoint",
                ),
                GeneratedArtifact(
                    path="/login",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b"<html><body>Login</body></html>",
                    status_code=200,
                    source_artifact_id="login",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/_flow/admin/unused",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b"<html><body>Unused</body></html>",
                    status_code=200,
                    source_artifact_id="unused",
                    artifact_scope="dynamic_endpoint",
                ),
            ],
            review_summary="approved",
            used_fallback=False,
            flow_descriptor=FlowDescriptor(
                rules=[
                    FlowRule(
                        match_path="/admin",
                        condition=FlowCondition(missing_cookie="session_token"),
                        response=FlowResponse(redirect_to="/login", status_code=302),
                        priority=10,
                    )
                ]
            ),
        )

        validate_bundle(bundle, request, runtime_config)
        diagnostics = diagnose_flow_reachability(bundle)

        self.assertIn("flow artifact /_flow/admin/unused is not served by any flow rule", diagnostics)

    def test_finalize_bundle_persists_flow_reachability_diagnostics(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(enable_scripted_flows=True, max_bundle_artifacts=3))
        request = ensure_generation_request("example.com", "/admin", {"index_page": "/index.html"})
        bundle = GeneratedBundle(
            primary_path="/admin",
            artifacts=[
                GeneratedArtifact(
                    path="/admin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b"<html><body>Admin</body></html>",
                    status_code=200,
                    source_artifact_id="admin",
                    artifact_scope="dynamic_endpoint",
                ),
                GeneratedArtifact(
                    path="/login",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b"<html><body>Login</body></html>",
                    status_code=200,
                    source_artifact_id="login",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/_flow/admin/unused",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b"<html><body>Unused</body></html>",
                    status_code=200,
                    source_artifact_id="unused",
                    artifact_scope="dynamic_endpoint",
                ),
            ],
            review_summary="approved",
            used_fallback=False,
            flow_descriptor=FlowDescriptor(
                rules=[
                    FlowRule(
                        match_path="/admin",
                        condition=FlowCondition(missing_cookie="session_token"),
                        response=FlowResponse(redirect_to="/login", status_code=302),
                        priority=10,
                    )
                ]
            ),
        )

        result = self.loop.run_until_complete(
            generator._finalize_bundle({"request": request, "generated_bundle": bundle})
        )
        finalized = result["generated_bundle"]

        self.assertFalse(finalized.used_fallback)
        self.assertIn("FLOW_REACHABILITY_WARNING", finalized.review_summary)
        self.assertIn("/_flow/admin/unused", finalized.review_summary)


    def test_finalize_bundle_keeps_artifact_count_overflow_when_bytes_fit(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(max_bundle_artifacts=2))
        request = ensure_generation_request("example.com", "/admin", {"index_page": "/index.html"})
        bundle = GeneratedBundle(
            primary_path="/admin",
            artifacts=[
                GeneratedArtifact(
                    path="/admin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><head><link rel="stylesheet" href="/app.css"></head><body>Admin</body></html>',
                    status_code=200,
                    source_artifact_id="admin",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/app.css",
                    kind="stylesheet",
                    headers=[{"Content-Type": "text/css"}],
                    body_bytes=b"body { color: #111; }",
                    status_code=200,
                    source_artifact_id="app-css",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/robots.txt",
                    kind="robots_txt",
                    headers=[{"Content-Type": "text/plain"}],
                    body_bytes=b"User-agent: *\nDisallow:\n",
                    status_code=200,
                    source_artifact_id="robots",
                    artifact_scope="static_file",
                ),
            ],
            review_summary="approved",
            used_fallback=False,
        )

        result = self.loop.run_until_complete(
            generator._finalize_bundle({"request": request, "generated_bundle": bundle})
        )
        finalized = result["generated_bundle"]

        self.assertEqual(
            [artifact.path for artifact in finalized.artifacts],
            ["/admin", "/app.css", "/robots.txt"],
        )
        self.assertFalse(
            any(
                diagnostic["category"] == "trimmed_to_byte_limit"
                for diagnostic in finalized.generation_diagnostics
            )
        )
        validate_bundle(finalized, request, generator.runtime_config)

    def test_finalize_bundle_trims_optional_artifacts_to_byte_limit(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(max_bundle_bytes=1024))
        request = ensure_generation_request("example.com", "/admin", {"index_page": "/index.html"})
        bundle = GeneratedBundle(
            primary_path="/admin",
            artifacts=[
                GeneratedArtifact(
                    path="/admin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><head><link rel="stylesheet" href="/app.css"></head><body>A</body></html>',
                    status_code=200,
                    source_artifact_id="admin",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/app.css",
                    kind="stylesheet",
                    headers=[{"Content-Type": "text/css"}],
                    body_bytes=b"body{color:#111}",
                    status_code=200,
                    source_artifact_id="app-css",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/robots.txt",
                    kind="robots_txt",
                    headers=[{"Content-Type": "text/plain"}],
                    body_bytes=(b"User-agent: *\nDisallow: /private\nSitemap: /sitemap.xml\n" * 32),
                    status_code=200,
                    source_artifact_id="robots",
                    artifact_scope="static_file",
                ),
            ],
            review_summary="approved",
            used_fallback=False,
        )

        result = self.loop.run_until_complete(
            generator._finalize_bundle({"request": request, "generated_bundle": bundle})
        )
        finalized = result["generated_bundle"]

        self.assertEqual([artifact.path for artifact in finalized.artifacts], ["/admin", "/app.css"])
        self.assertTrue(
            any(
                diagnostic["category"] == "trimmed_to_byte_limit"
                for diagnostic in finalized.generation_diagnostics
            )
        )
        validate_bundle(finalized, request, generator.runtime_config)


    def test_flow_designer_does_not_add_missing_cookie_guard_to_login_page(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(enable_scripted_flows=True))
        request = ensure_generation_request(
            "example.com",
            "/boaform/admin/formLogin",
            {"index_page": "/index.html"},
        )
        resource_plan = ResourcePlan(
            primary_path="/boaform/admin/formLogin",
            theme_summary="Legacy router admin portal",
            artifacts=[
                PlannedArtifact(
                    artifact_id="login-page",
                    path="/admin/login.asp",
                    kind="html_page",
                    purpose="Login page",
                    dynamic_candidate=True,
                ),
                PlannedArtifact(
                    artifact_id="login-handler",
                    path="/boaform/admin/formLogin",
                    kind="html_page",
                    purpose="Form handler",
                    artifact_scope="dynamic_endpoint",
                    dynamic_candidate=True,
                ),
            ],
            bundle_budget_count=4,
            bundle_budget_bytes=16384,
            static_only=False,
            review_focus=["flow"],
        )
        bundle = GeneratedBundle(
            primary_path="/boaform/admin/formLogin",
            artifacts=[
                GeneratedArtifact(
                    path="/admin/login.asp",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=(
                        b'<html><body><form method="post" action="/boaform/admin/formLogin">'
                        b'<input name="username"><input name="password" type="password">'
                        b'</form></body></html>'
                    ),
                    status_code=200,
                    source_artifact_id="login-page",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/boaform/admin/formLogin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><body>handler</body></html>',
                    status_code=200,
                    source_artifact_id="login-handler",
                    artifact_scope="dynamic_endpoint",
                ),
            ],
            review_summary="pending",
            used_fallback=False,
        )
        state = {
            "request": request,
            "resource_plan": resource_plan,
            "generated_bundle": bundle.model_dump(mode="json"),
        }

        result = self.loop.run_until_complete(generator._flow_designer_node(state))
        descriptor = FlowDescriptor.model_validate(result["flow_descriptor"])
        result_bundle = GeneratedBundle.model_validate(result["generated_bundle"])
        self.assertIsNotNone(result_bundle.flow_descriptor)

        self.assertFalse(
            any(
                rule.match_path == "/admin/login.asp"
                and rule.condition is not None
                and rule.condition.missing_cookie == "session_token"
                for rule in descriptor.rules
            )
        )

    def test_flow_designer_output_validates_before_review(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(enable_scripted_flows=True))
        request = ensure_generation_request(
            "example.com",
            "/boaform/admin/formLogin",
            {"index_page": "/index.html"},
        )
        expert_spec = self.loop.run_until_complete(generator._heuristic_expert_spec(request))
        resource_plan = ResourcePlan(
            primary_path="/boaform/admin/formLogin",
            theme_summary="Legacy router admin portal",
            artifacts=[
                PlannedArtifact(
                    artifact_id="login-page",
                    path="/admin/login.asp",
                    kind="html_page",
                    purpose="Login page",
                    dynamic_candidate=True,
                ),
                PlannedArtifact(
                    artifact_id="login-handler",
                    path="/boaform/admin/formLogin",
                    kind="html_page",
                    purpose="Form handler",
                    artifact_scope="dynamic_endpoint",
                    dynamic_candidate=True,
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16384,
            static_only=False,
            review_focus=["flow"],
        )
        bundle = GeneratedBundle(
            primary_path="/boaform/admin/formLogin",
            artifacts=[
                GeneratedArtifact(
                    path="/admin/login.asp",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=(
                        b'<html><body><form method="post" action="/boaform/admin/formLogin">'
                        b'<input name="username"><input name="password" type="password">'
                        b'</form></body></html>'
                    ),
                    status_code=200,
                    source_artifact_id="login-page",
                    artifact_scope="static_file",
                ),
                GeneratedArtifact(
                    path="/boaform/admin/formLogin",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b'<html><body>handler</body></html>',
                    status_code=200,
                    source_artifact_id="login-handler",
                    artifact_scope="dynamic_endpoint",
                ),
            ],
            review_summary="pending",
            used_fallback=False,
        )
        state = {
            "request": request,
            "expert_spec": expert_spec,
            "resource_plan": resource_plan,
            "generated_bundle": bundle.model_dump(mode="json"),
            "errors": [],
        }

        flow_result = self.loop.run_until_complete(generator._flow_designer_node(state))
        review_result = self.loop.run_until_complete(
            generator._review_node({**state, **flow_result})
        )

        self.assertEqual(review_result["review_decision"].decision, "approve")

    def test_optional_first_class_flow_variants_map_to_rules(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(enable_scripted_flows=True))

        mfa_rule = generator._variant_to_rule(
            "/admin/login",
            "mfa-required",
            "/_flow/admin-login/mfa-required",
            {"/admin/login", "/_flow/admin-login/mfa-required"},
        )
        expired_rule = generator._variant_to_rule(
            "/admin/panel",
            "session-expired",
            "/_flow/admin-panel/session-expired",
            {"/admin/panel", "/_flow/admin-panel/session-expired"},
        )

        self.assertEqual(mfa_rule.condition.method, "POST")
        self.assertEqual(mfa_rule.response.artifact_path, "/_flow/admin-login/mfa-required")
        self.assertEqual(mfa_rule.response.set_cookie, {"session_stage": "mfa_required"})
        self.assertEqual(expired_rule.condition.missing_cookie, "session_token")
        self.assertEqual(expired_rule.response.artifact_path, "/_flow/admin-panel/session-expired")

    def test_explicit_flow_metadata_builds_non_login_rule(self):
        generator = NoModelGenerator(runtime_config=self._runtime_config(enable_scripted_flows=True))
        request = ensure_generation_request(
            "example.com",
            "/manager/html",
            {"index_page": "/index.html"},
        )
        resource_plan = ResourcePlan(
            primary_path="/manager/html",
            theme_summary="Management console",
            artifacts=[
                PlannedArtifact(
                    artifact_id="manager-page",
                    path="/manager/html",
                    kind="html_page",
                    purpose="Management console landing page",
                ),
                PlannedArtifact(
                    artifact_id="manager-auth-required",
                    path="/_flow/manager-html/auth-required",
                    kind="html_page",
                    purpose="Authorization required response for missing credentials",
                    flow_match_path="/manager/html",
                    flow_condition=FlowCondition(missing_header="Authorization"),
                    flow_response=FlowResponse(
                        artifact_path="/_flow/manager-html/auth-required",
                        status_code=401,
                        headers=[{"WWW-Authenticate": "Basic realm=\"Management Console\""}],
                    ),
                    flow_priority=20,
                ),
            ],
            bundle_budget_count=2,
            bundle_budget_bytes=16384,
            static_only=False,
            coherence_facts={"service_name": "Management Console"},
        )
        bundle = GeneratedBundle(
            primary_path="/manager/html",
            artifacts=[
                GeneratedArtifact(
                    path="/manager/html",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b"<html><body>Manager</body></html>",
                    status_code=200,
                    source_artifact_id="manager-page",
                ),
                GeneratedArtifact(
                    path="/_flow/manager-html/auth-required",
                    kind="html_page",
                    headers=[{"Content-Type": "text/html"}],
                    body_bytes=b"<html><body>Authorization required</body></html>",
                    status_code=200,
                    source_artifact_id="manager-auth-required",
                ),
            ],
            review_summary="pending",
            used_fallback=False,
        )

        validate_plan(resource_plan, request, generator.runtime_config)
        result = self.loop.run_until_complete(
            generator._flow_designer_node(
                {
                    "request": request,
                    "resource_plan": resource_plan,
                    "generated_bundle": bundle.model_dump(mode="json"),
                }
            )
        )
        descriptor = FlowDescriptor.model_validate(result["flow_descriptor"])

        self.assertEqual(len(descriptor.rules), 1)
        rule = descriptor.rules[0]
        self.assertEqual(rule.match_path, "/manager/html")
        self.assertEqual(rule.condition.missing_header, "Authorization")
        self.assertEqual(rule.response.status_code, 401)
        self.assertEqual(rule.response.artifact_path, "/_flow/manager-html/auth-required")

    def test_flow_evaluator_matches_headers_query_and_post_fields(self):
        evaluator = FlowEvaluator()
        evaluator.load_from_dict(
            "/request-shape",
            {
                "rules": [
                    {
                        "match_path": "/manager/html",
                        "condition": {"missing_header": "Authorization"},
                        "response": {
                            "artifact_path": "/_flow/manager-html/auth-required",
                            "status_code": 401,
                            "headers": [{"WWW-Authenticate": "Basic realm=\"Management Console\""}],
                        },
                        "priority": 30,
                    },
                    {
                        "match_path": "/shell",
                        "condition": {"query_has": ["cmd"], "header_contains": {"User-Agent": "curl"}},
                        "response": {"artifact_path": "/_flow/shell/command-response", "status_code": 200},
                        "priority": 20,
                    },
                    {
                        "match_path": "/upload",
                        "condition": {"method": "POST", "post_has": ["filename"], "post_contains": {"filename": ".php"}},
                        "response": {"artifact_path": "/_flow/upload/upload-denied", "status_code": 403},
                        "priority": 10,
                    },
                ]
            },
        )

        class FakeSession:
            paths = []
            cookies = {}

        auth_required = evaluator.evaluate(FakeSession(), "/manager/html", {"method": "GET", "headers": {}})
        authorized = evaluator.evaluate(FakeSession(), "/manager/html", {"method": "GET", "headers": {"authorization": "Basic abc"}})
        command = evaluator.evaluate(FakeSession(), "/shell", {"method": "GET", "path": "/shell?cmd=id", "headers": {"user-agent": "curl/8.0"}})
        upload = evaluator.evaluate(FakeSession(), "/upload", {"method": "POST", "post_data": {"filename": "cmd.php"}})

        self.assertEqual(auth_required.status_code, 401)
        self.assertEqual(auth_required.headers["WWW-Authenticate"], "Basic realm=\"Management Console\"")
        self.assertFalse(authorized.matched)
        self.assertEqual(command.artifact_path, "/_flow/shell/command-response")
        self.assertEqual(upload.artifact_path, "/_flow/upload/upload-denied")


    def test_flow_evaluator_enforces_three_attempt_loop_with_one_minute_lockout(self):
        evaluator = FlowEvaluator()
        evaluator.load_from_dict(
            "/boaform/admin/formLogin",
            {
                "rules": [
                    {
                        "match_path": "/boaform/admin/formLogin",
                        "condition": {
                            "method": "POST",
                            "min_prior_post_count_to_path": 3,
                            "lockout_window_seconds": 60,
                            "lockout_active": True,
                        },
                        "response": {"artifact_path": "/_flow/boaform-admin-formLogin/post-locked", "status_code": 200},
                        "priority": 10,
                    },
                    {
                        "match_path": "/boaform/admin/formLogin",
                        "condition": {"method": "POST"},
                        "response": {"artifact_path": "/_flow/boaform-admin-formLogin/post-fail", "status_code": 200},
                        "priority": 5,
                    },
                ]
            },
        )

        class FakeSession:
            def __init__(self, paths):
                self.paths = paths
                self.cookies = {}

        base_ts = 1000.0
        first = evaluator.evaluate(FakeSession([{"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts}]), "/boaform/admin/formLogin", {"method": "POST"})
        second = evaluator.evaluate(FakeSession([
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 1},
        ]), "/boaform/admin/formLogin", {"method": "POST"})
        third = evaluator.evaluate(FakeSession([
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 1},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 2},
        ]), "/boaform/admin/formLogin", {"method": "POST"})
        locked = evaluator.evaluate(FakeSession([
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 1},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 2},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 10},
        ]), "/boaform/admin/formLogin", {"method": "POST"})
        reset = evaluator.evaluate(FakeSession([
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 1},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 2},
            {"path": "/boaform/admin/formLogin", "method": "POST", "timestamp": base_ts + 65},
        ]), "/boaform/admin/formLogin", {"method": "POST"})

        self.assertEqual(first.artifact_path, "/_flow/boaform-admin-formLogin/post-fail")
        self.assertEqual(second.artifact_path, "/_flow/boaform-admin-formLogin/post-fail")
        self.assertEqual(third.artifact_path, "/_flow/boaform-admin-formLogin/post-fail")
        self.assertEqual(locked.artifact_path, "/_flow/boaform-admin-formLogin/post-locked")
        self.assertEqual(reset.artifact_path, "/_flow/boaform-admin-formLogin/post-fail")
