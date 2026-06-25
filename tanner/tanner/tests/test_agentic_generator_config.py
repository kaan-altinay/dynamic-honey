import unittest
from unittest import mock

from tanner.generator.agentic.config import load_runtime_config
from tanner.generator.agentic.model_factory import build_role_model


class TestAgenticGeneratorConfig(unittest.TestCase):
    @staticmethod
    def _config_get(section, value):
        values = {
            ("GENERATOR", "backend"): "agentic",
            ("GENERATOR", "max_review_loops"): 3,
            ("GENERATOR", "max_bundle_artifacts"): 5,
            ("GENERATOR", "max_bundle_bytes"): 99_999,
            ("GENERATOR", "checkpoint_path"): "/tmp/test-checkpoints.sqlite",
            ("GENERATOR", "graph_recursion_limit"): 180,
            ("GENERATOR", "review_log_path"): "/tmp/test-review-log.json",
            ("GENERATOR", "enable_live_research"): False,
            ("GENERATOR", "max_tool_response_chars"): 1024,
            ("GENERATOR", "max_command_output_chars"): 2048,
            ("GENERATOR", "command_timeout"): 7,
            ("GENERATOR", "max_concurrent_model_calls"): 1,
            ("GENERATOR", "inter_call_delay_seconds"): 9.5,
            ("GENERATOR", "max_rate_limit_retries"): 4,
            ("GENERATOR", "default_rate_limit_backoff_seconds"): 13.0,
            ("GENERATOR", "max_length_limit_retries"): 5,
            ("GENERATOR", "length_retry_token_increase"): 900,
            ("GENERATOR", "max_length_retry_tokens"): 7200,
            ("GENERATOR", "role_defaults"): {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 600,
                "timeout": 11,
                "max_retries": 4,
            },
            ("GENERATOR", "roles"): {
                "review": {"provider": "anthropic", "model": "claude-3-5-haiku-latest", "temperature": 0.05},
                "coder": {"provider": "ollama", "model": "qwen2.5:14b-instruct", "max_tokens": 900},
            },
        }
        return values.get((section, value))

    def test_load_runtime_config_reads_role_overrides(self):
        with mock.patch("tanner.generator.agentic.config.TannerConfig.get", side_effect=self._config_get):
            runtime_config = load_runtime_config()

        self.assertEqual(runtime_config.backend, "agentic")
        self.assertEqual(runtime_config.max_review_loops, 3)
        self.assertEqual(runtime_config.max_bundle_artifacts, 5)
        self.assertFalse(runtime_config.enable_live_research)
        self.assertEqual(runtime_config.graph_recursion_limit, 180)
        self.assertEqual(runtime_config.review_log_path, "/tmp/test-review-log.json")
        self.assertEqual(runtime_config.max_concurrent_model_calls, 1)
        self.assertEqual(runtime_config.inter_call_delay_seconds, 9.5)
        self.assertEqual(runtime_config.max_rate_limit_retries, 4)
        self.assertEqual(runtime_config.default_rate_limit_backoff_seconds, 13.0)
        self.assertEqual(runtime_config.max_length_limit_retries, 5)
        self.assertEqual(runtime_config.length_retry_token_increase, 900)
        self.assertEqual(runtime_config.max_length_retry_tokens, 7200)
        self.assertEqual(runtime_config.role_config("expert").provider, "openai")
        self.assertEqual(runtime_config.role_config("expert").model, "gpt-4o-mini")
        self.assertEqual(runtime_config.role_config("coder").provider, "ollama")
        self.assertEqual(runtime_config.role_config("coder").max_tokens, 900)
        self.assertEqual(runtime_config.role_config("review").provider, "anthropic")
        self.assertEqual(runtime_config.role_config("review").temperature, 0.05)

    def test_build_role_model_uses_provider_agnostic_init(self):
        with mock.patch("tanner.generator.agentic.config.TannerConfig.get", side_effect=self._config_get):
            runtime_config = load_runtime_config()

        with mock.patch("tanner.generator.agentic.model_factory.init_chat_model") as init_model:
            sentinel = object()
            init_model.return_value = sentinel
            model = build_role_model("review", runtime_config)

        self.assertIs(model, sentinel)
        init_model.assert_called_once_with(
            "claude-3-5-haiku-latest",
            model_provider="anthropic",
            temperature=0.05,
            timeout=11,
            max_tokens=600,
            max_retries=4,
        )


class TestAgenticGeneratorV2Overrides(unittest.TestCase):
    @staticmethod
    def _config_get(section, value):
        values = {
            ("GENERATOR", "backend"): "agentic",
            ("GENERATOR", "enable_scripted_flows"): True,
            ("GENERATOR", "max_review_loops"): 1,
            ("GENERATOR", "max_design_validation_loops"): 2,
            ("GENERATOR", "max_bundle_artifacts"): 8,
            ("GENERATOR", "max_bundle_bytes"): 2_000_000,
            ("GENERATOR", "role_defaults"): {
                "provider": "openai",
                "model": "gpt-5.4",
                "temperature": 0.2,
                "max_tokens": 900,
                "timeout": 120,
                "max_retries": 1,
            },
            ("GENERATOR", "roles"): {
                "expert": {"temperature": 0.0, "max_tokens": 1800},
                "design": {"temperature": 0.0, "max_tokens": 2800},
                "coder": {"temperature": 0.1, "max_tokens": 1600},
                "review": {"temperature": 0.0, "max_tokens": 1600},
            },
            ("GENERATOR", "v2_overrides"): {
                "max_review_loops": 1,
                "max_design_validation_loops": 2,
                "max_bundle_artifacts": 8,
                "max_bundle_bytes": 2_000_000,
                "roles": {
                    "expert": {"temperature": 0.0, "max_tokens": 1800},
                    "design": {"temperature": 0.1, "max_tokens": 2400},
                    "coder": {"temperature": 0.1, "max_tokens": 1600},
                    "review": {"temperature": 0.0, "max_tokens": 1600},
                },
            },
        }
        return values.get((section, value))

    def test_scripted_flows_enable_v2_tuned_overrides(self):
        with mock.patch("tanner.generator.agentic.config.TannerConfig.get", side_effect=self._config_get):
            runtime_config = load_runtime_config()
        self.assertTrue(runtime_config.enable_scripted_flows)
        self.assertEqual(runtime_config.max_bundle_artifacts, 8)
        self.assertEqual(runtime_config.max_bundle_bytes, 2_000_000)
        self.assertEqual(runtime_config.max_review_loops, 1)
        self.assertEqual(runtime_config.max_design_validation_loops, 2)
        self.assertEqual(runtime_config.role_config("expert").temperature, 0.0)
        self.assertEqual(runtime_config.role_config("design").temperature, 0.1)
        self.assertEqual(runtime_config.role_config("design").max_tokens, 2400)
        self.assertEqual(runtime_config.role_config("coder").temperature, 0.1)
        self.assertEqual(runtime_config.role_config("coder").max_tokens, 1600)
        self.assertEqual(runtime_config.role_config("review").temperature, 0.0)
