import asyncio
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from tanner.generator.agentic.workflow import AgenticBundleGenerator
from tanner.generator.agentic.models import GeneratorRoleConfig, GeneratorRuntimeConfig


def role(model: str, temperature: float, max_tokens: int) -> GeneratorRoleConfig:
    return GeneratorRoleConfig(
        provider="openai",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=1,
    )


def dump(value):
    if hasattr(value, "model_dump_json"):
        try:
            return value.model_dump_json(indent=2)
        except Exception:
            pass
    return json.dumps(value, indent=2, default=str)


class TracingGenerator(AgenticBundleGenerator):
    async def _invoke_structured(self, role_name: str, schema, messages):
        print(f"\n=== INVOKING {role_name.upper()} ({schema.__name__}) ===")
        try:
            result = await super()._invoke_structured(role_name, schema, messages)
            print(f"=== MODEL SUCCESS: {role_name.upper()} ===")
            print(dump(result))
            return result
        except Exception:
            print(f"=== MODEL FAILURE: {role_name.upper()} ===")
            traceback.print_exc()
            raise


async def main():
    model_name = "gpt-5.4"

    cfg = GeneratorRuntimeConfig(
        backend="agentic",
        max_review_loops=1,
        max_bundle_artifacts=8,
        max_bundle_bytes=2_000_000,
        checkpoint_path="/tmp/tanner-agentic-browser-preview.sqlite",
        graph_recursion_limit=200,
        review_log_path="/tmp/tanner-agentic-review-log.json",
        enable_live_research=True,
        max_tool_response_chars=4000,
        max_command_output_chars=2000,
        command_timeout=2,

        max_concurrent_model_calls=1,
        inter_call_delay_seconds=12.5,
        max_rate_limit_retries=3,
        default_rate_limit_backoff_seconds=12.0,

        max_length_limit_retries=3,
        length_retry_token_increase=800,
        max_length_retry_tokens=6000,

        roles={
            "expert": role(model_name, 0.0, 1800),
            "design": role(model_name, 0.0, 2800),
            "coder": role(model_name, 0.1, 1600),
            "review": role(model_name, 0.0, 1600),
        },
    )

    generator = TracingGenerator(runtime_config=cfg)

    path = "/tr064dev.xml"
    bundle = await generator.generate_bundle(
        host="example.com",
        path=path,
        site_profile={"index_page": "/index.html"},
    )

    print("\n=== FINAL BUNDLE ===")
    print("primary_path:", bundle.primary_path)
    print("used_fallback:", bundle.used_fallback)
    print("review_summary:", bundle.review_summary)
    print("artifact_count:", len(bundle.artifacts))

    artifact_map = {artifact.path: artifact for artifact in bundle.artifacts}
    index_page = "/index.html"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            req_path = self.path.split("?", 1)[0]
            if req_path == "/":
                req_path = bundle.primary_path

            artifact = artifact_map.get(req_path)

            # Allow previewing bundles that link back to the baseline index page.
            if artifact is None and req_path == index_page:
                self.send_response(302)
                self.send_header("Location", bundle.primary_path)
                self.end_headers()
                return

            if artifact is None:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            self.send_response(artifact.status_code)
            content_type = None
            for header in artifact.headers:
                if isinstance(header, dict):
                    for key, value in header.items():
                        if key.lower() == "content-type":
                            content_type = value
                        self.send_header(key, value)
            if content_type is None:
                self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(artifact.body_bytes)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("\n=== PREVIEW SERVER ===")
    print("Open this URL in your LOCAL browser:")
    print(f"http://127.0.0.1:8765{bundle.primary_path}")
    print("\nBecause of SSH port forwarding, the above localhost URL should work on your own machine.")
    print("Press Ctrl+C in this VM terminal to stop the preview server.")

    try:
        await asyncio.Event().wait()
    finally:
        server.shutdown()
        server.server_close()


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nPreview stopped.")