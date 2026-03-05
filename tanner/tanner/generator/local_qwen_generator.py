import asyncio
import html
import json
import logging

import aiohttp

from tanner.config import TannerConfig
from tanner.generator.base_generator import BaseGenerator


class LocalQwenGenerator(BaseGenerator):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.endpoint = self._as_string(self._get_config("endpoint", "http://127.0.0.1:8000/v1/chat/completions"), "http://127.0.0.1:8000/v1/chat/completions")
        self.model = self._as_string(self._get_config("model", "qwen-local"), "qwen-local")
        self.timeout = self._as_int(self._get_config("timeout", 60), 60, min_value=1)
        self.retries = self._as_int(self._get_config("retries", 2), 2, min_value=0)
        self.temperature = self._as_float(self._get_config("temperature", 0.15), 0.15)
        self.max_tokens = self._as_int(self._get_config("max_tokens", 700), 700, min_value=64)

    @staticmethod
    def _get_config(value, default):
        try:
            configured_value = TannerConfig.get("GENERATOR", value)
        except KeyError:
            return default
        return configured_value if configured_value is not None else default

    @staticmethod
    def _as_string(value, default):
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default

    @staticmethod
    def _as_int(value, default, min_value=0):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default

        if parsed < min_value:
            return default
        return parsed

    @staticmethod
    def _as_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_path(path):
        if not isinstance(path, str) or not path.strip():
            return "/"
        normalized = path.strip()
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        return normalized

    @staticmethod
    def _infer_complexity(path, site_profile):
        normalized_path = LocalQwenGenerator._normalize_path(path)
        complexity = "simple"

        if "?" in normalized_path or normalized_path.count("/") > 3:
            complexity = "standard"

        lowered = normalized_path.lower()
        if any(marker in lowered for marker in ["/admin", "/dashboard", "/settings", "/account", "/api"]):
            complexity = "complex"

        if isinstance(site_profile, dict):
            candidates = site_profile.get("candidates")
            if isinstance(candidates, list):
                if len(candidates) >= 6:
                    complexity = "complex"
                elif len(candidates) >= 3 and complexity == "simple":
                    complexity = "standard"

        return complexity

    @staticmethod
    def _response_schema():
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "status_code": {"type": "integer"},
                "headers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["name", "value"],
                        "additionalProperties": False,
                    },
                },
                "body_html": {"type": "string"},
            },
            "required": ["path", "status_code", "headers", "body_html"],
            "additionalProperties": False,
        }

    def _build_messages(self, host, path, site_profile):
        normalized_path = self._normalize_path(path)
        complexity = self._infer_complexity(normalized_path, site_profile)
        profile = {
            "host": host or "unknown.local",
            "path": normalized_path,
            "complexity": complexity,
            "index_page": site_profile.get("index_page") if isinstance(site_profile, dict) else None,
            "candidate_count": len(site_profile.get("candidates", [])) if isinstance(site_profile, dict) and isinstance(site_profile.get("candidates"), list) else 0,
        }

        system_prompt = (
            "Return valid JSON only with keys path,status_code,headers,body_html. "
            "Never return markdown. headers must be an array of {name,value} objects. "
            "body_html must be a full HTML document."
        )
        user_prompt = (
            "Generate a believable web page using this profile JSON: {profile}. "
            "Use status_code 200 and include a Content-Type header for text/html; charset=utf-8."
        ).format(profile=json.dumps(profile, sort_keys=True))

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    async def _request_completion(self, payload):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.endpoint, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    @staticmethod
    def _extract_content(response_data):
        choices = response_data.get("choices") if isinstance(response_data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("Missing completion choices")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise ValueError("Missing completion message")

        content = message.get("content")
        if isinstance(content, dict):
            return content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            content = "".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Completion content is empty")

        return json.loads(content)

    @staticmethod
    def _normalize_headers(headers):
        normalized = []

        if isinstance(headers, dict):
            for key, value in headers.items():
                if isinstance(key, str) and isinstance(value, str):
                    normalized.append({key: value})
        elif isinstance(headers, list):
            for header in headers:
                if not isinstance(header, dict):
                    continue

                name = header.get("name")
                value = header.get("value")
                if isinstance(name, str) and isinstance(value, str):
                    normalized.append({name: value})
                    continue

                for key, alt_value in header.items():
                    if isinstance(key, str) and isinstance(alt_value, str):
                        normalized.append({key: alt_value})

        has_content_type = any(
            isinstance(header, dict)
            and any(isinstance(key, str) and key.lower() == "content-type" for key in header.keys())
            for header in normalized
        )
        if not has_content_type:
            normalized.append({"Content-Type": "text/html; charset=utf-8"})

        return normalized

    def _build_payload(self, host, path, site_profile):
        return {
            "model": self.model,
            "messages": self._build_messages(host, path, site_profile),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object", "schema": self._response_schema()},
        }

    def _build_result(self, model_output, requested_path):
        page_path = model_output.get("path")
        if not isinstance(page_path, str) or not page_path.strip():
            page_path = self._normalize_path(requested_path)
        page_path = self._normalize_path(page_path)

        status_code = model_output.get("status_code")
        if not isinstance(status_code, int):
            raise ValueError("status_code must be an integer")

        body_html = model_output.get("body_html")
        if not isinstance(body_html, str) or not body_html.strip():
            raise ValueError("body_html must be a non-empty string")
        if len(body_html.encode("utf-8")) > 200_000:
            raise ValueError("body_html exceeds maximum allowed size")

        headers = self._normalize_headers(model_output.get("headers"))

        return {
            "path": page_path,
            "headers": headers,
            "body_bytes": body_html.encode("utf-8"),
        }

    def _fallback_page(self, host, path, error):
        safe_host = html.escape(host or "unknown.local")
        safe_path = html.escape(self._normalize_path(path))
        fallback_html = (
            "<!DOCTYPE html>"
            "<html lang=\"en\">"
            "<head><meta charset=\"utf-8\"><title>Page Unavailable</title></head>"
            "<body>"
            "<h1>Service Temporarily Unavailable</h1>"
            "<p>Host: {host}</p>"
            "<p>Path: {path}</p>"
            "</body></html>"
        ).format(host=safe_host, path=safe_path)
        self.logger.warning("Using fallback generated page due to LLM failure: %s", error)

        return {
            "path": self._normalize_path(path),
            "headers": [{"Content-Type": "text/html; charset=utf-8"}],
            "body_bytes": fallback_html.encode("utf-8"),
        }

    async def generate_page(self, host, path, site_profile):
        payload = self._build_payload(host, path, site_profile)
        last_error = None

        for _ in range(self.retries + 1):
            try:
                response_data = await self._request_completion(payload)
                model_output = self._extract_content(response_data)
                return self._build_result(model_output, path)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
                last_error = error
                self.logger.warning("LocalQwenGenerator attempt failed: %s", error)

        return self._fallback_page(host, path, last_error)
