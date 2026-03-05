import asyncio
import json
import unittest
from unittest import mock

import aiohttp

from tanner.generator.local_qwen_generator import LocalQwenGenerator


class TestLocalQwenGenerator(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(None)

        self._config = {
            ("GENERATOR", "endpoint"): "http://127.0.0.1:8000/v1/chat/completions",
            ("GENERATOR", "model"): "qwen-local",
            ("GENERATOR", "timeout"): 12,
            ("GENERATOR", "retries"): 1,
            ("GENERATOR", "temperature"): 0.1,
            ("GENERATOR", "max_tokens"): 512,
        }
        self.config_patcher = mock.patch(
            "tanner.generator.local_qwen_generator.TannerConfig.get", side_effect=self._mock_config_get
        )
        self.config_patcher.start()
        self.generator = LocalQwenGenerator()

    def tearDown(self):
        self.config_patcher.stop()
        self.loop.close()

    def _mock_config_get(self, section, value):
        return self._config[(section, value)]

    def test_generate_page_success(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "path": "/admin/login",
                                "status_code": 200,
                                "headers": [{"name": "X-Source", "value": "local-qwen"}],
                                "body_html": "<!DOCTYPE html><html><body><h1>Admin Login</h1></body></html>",
                            }
                        )
                    }
                }
            ]
        }

        with mock.patch.object(self.generator, "_request_completion", new=mock.AsyncMock(return_value=response_payload)):
            result = self.loop.run_until_complete(
                self.generator.generate_page(host="example.com", path="/admin/login", site_profile={"index_page": "/index.html"})
            )

        self.assertEqual(result["path"], "/admin/login")
        self.assertIsInstance(result["body_bytes"], bytes)
        self.assertIn(b"Admin Login", result["body_bytes"])
        self.assertIn({"X-Source": "local-qwen"}, result["headers"])
        self.assertIn({"Content-Type": "text/html; charset=utf-8"}, result["headers"])

    def test_generate_page_fallback_after_failures(self):
        with mock.patch.object(
            self.generator,
            "_request_completion",
            new=mock.AsyncMock(side_effect=aiohttp.ClientError("connect failed")),
        ):
            result = self.loop.run_until_complete(
                self.generator.generate_page(host="example.com", path="/status", site_profile={})
            )

        self.assertEqual(result["path"], "/status")
        self.assertEqual(result["headers"], [{"Content-Type": "text/html; charset=utf-8"}])
        self.assertIn(b"Service Temporarily Unavailable", result["body_bytes"])
