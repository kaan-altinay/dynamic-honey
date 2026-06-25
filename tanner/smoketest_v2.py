"""
V2 smoketest: generate a bundle with scripted flows enabled and serve it
interactively so flow rules (redirects, rewrites, cookie gating) fire in
the preview browser exactly as they would at runtime.

Usage:
    cd /home/kaan/dynamic-honey/tanner
    .venv/bin/python smoketest_v2.py [/optional/path]

Then open http://127.0.0.1:8765<path> in your local browser (via SSH port
forward).  Cookies and session history are tracked across requests so you
can walk through the full flow: GET login → POST credentials → redirect →
gated admin page.

SSH port forward (run on your local machine):
    ssh -L 8765:127.0.0.1:8765 proxmox-vm
"""

import argparse
import asyncio
import hashlib
import http.cookies
import json
import sys
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import unquote

from tanner.config import TannerConfig
from tanner.flow.flow_evaluator import FlowEvaluator, FlowMatchResult
from tanner.generator.agentic.config import load_runtime_config
from tanner.generator.agentic.models import FlowDescriptor, GeneratorRuntimeConfig
from tanner.generator.agentic.workflow import AgenticBundleGenerator

# This script always runs the V2 (scripted-flows) generator. Tuned values
# live in GENERATOR.v2_overrides in this file -- the same single source of
# truth merged by load_runtime_config() for the real server, kept in sync
# with tanner/tanner/data/config.yaml's GENERATOR.v2_overrides.
_V2_SMOKETEST_CONFIG_PATH = Path(__file__).resolve().parent / "tanner" / "data" / "config.v2-smoketest.yaml"


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize_path(path: str) -> str:
    normalized = (path or "").split("?", 1)[0]
    normalized = unquote(normalized)
    if not normalized:
        normalized = "/"
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized


def build_smoketest_config(model_name: str = "gpt-5.4") -> GeneratorRuntimeConfig:
    TannerConfig.set_config(str(_V2_SMOKETEST_CONFIG_PATH))
    cfg = load_runtime_config()
    return cfg.model_copy(
        update={
            "roles": {
                name: role_cfg.model_copy(update={"model": model_name})
                for name, role_cfg in cfg.roles.items()
            }
        }
    )


def save_bundle_to_snare_root(
    bundle,
    snare_root: str | Path,
    *,
    page_url: str = "example.com",
) -> Path:
    snare_root = Path(snare_root).resolve()
    page_dir = snare_root / "pages" / page_url
    page_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    for artifact in bundle.artifacts:
        artifact_path = _normalize_path(artifact.path)
        body = bytes(artifact.body_bytes)
        digest = hashlib.md5(body).hexdigest()
        (page_dir / digest).write_bytes(body)
        meta[artifact_path] = {
            "hash": digest,
            "headers": artifact.headers if isinstance(artifact.headers, list) else [],
        }

    (page_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    if bundle.flow_descriptor is not None:
        flows = {bundle.primary_path: bundle.flow_descriptor.model_dump(mode="json")}
        (page_dir / "flows.json").write_text(json.dumps(flows, indent=2, sort_keys=True) + "\n")

    summary = {
        "primary_path": bundle.primary_path,
        "used_fallback": bundle.used_fallback,
        "review_summary": bundle.review_summary,
        "artifact_count": len(bundle.artifacts),
        "artifacts": [
            {
                "path": artifact.path,
                "kind": artifact.kind,
                "bytes": len(bytes(artifact.body_bytes)),
                "headers": artifact.headers,
            }
            for artifact in bundle.artifacts
        ],
        "flow_descriptor": bundle.flow_descriptor.model_dump(mode="json") if bundle.flow_descriptor is not None else None,
    }
    (snare_root / "bundle_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return snare_root
def dump(value):
    if hasattr(value, "model_dump_json"):
        try:
            return value.model_dump_json(indent=2)
        except Exception:
            pass
    return json.dumps(value, indent=2, default=str)


def print_flow_descriptor(descriptor: FlowDescriptor) -> None:
    print(f"\n  {len(descriptor.rules)} rule(s):")
    for i, rule in enumerate(sorted(descriptor.rules, key=lambda r: -r.priority)):
        cond = rule.condition
        resp = rule.response
        cond_str = "always"
        if cond:
            parts = []
            if cond.method:
                parts.append(f"method={cond.method}")
            if cond.requires_cookie:
                parts.append(f"cookie '{cond.requires_cookie}' present")
            if cond.missing_cookie:
                parts.append(f"cookie '{cond.missing_cookie}' absent")
            if cond.requires_prev_path:
                parts.append(f"prev={cond.requires_prev_path}")
            if cond.min_post_count_to_path:
                parts.append(f"post_count>={cond.min_post_count_to_path}")
            if cond.min_prior_post_count_to_path:
                parts.append(f"prior_post_count>={cond.min_prior_post_count_to_path}")
            if cond.lockout_window_seconds:
                parts.append(f"lockout_window={cond.lockout_window_seconds}s")
            if cond.lockout_active is True:
                parts.append("lockout=active")
            elif cond.lockout_active is False:
                parts.append("lockout=inactive")
            cond_str = ", ".join(parts) if parts else "always"
        action = []
        if resp.redirect_to:
            action.append(f"302 -> {resp.redirect_to}")
        if resp.artifact_path:
            action.append(f"serve {resp.artifact_path}")
        if resp.set_cookie:
            action.append(f"set-cookie {resp.set_cookie}")
        if resp.clear_cookie:
            action.append(f"clear-cookie {resp.clear_cookie}")
        print(f"  [{i+1}] priority={rule.priority}  {rule.match_path}  ({cond_str})")
        print(f"       → {', '.join(action)}")


# ── tracing subclass ──────────────────────────────────────────────────────────

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


# ── session store ─────────────────────────────────────────────────────────────
# Keyed by smoketest_sid cookie.  Tracks path history and cookies for each
# browser session so flow conditions (prev_path, post_count, cookie checks)
# evaluate correctly across requests.

_sessions: dict[str, dict] = {}


def _get_or_create_session(sid: str | None) -> tuple[str, dict]:
    if sid and sid in _sessions:
        return sid, _sessions[sid]
    new_sid = uuid.uuid4().hex
    _sessions[new_sid] = {"cookies": {}, "paths": []}
    return new_sid, _sessions[new_sid]


def _parse_cookies(cookie_header: str | None) -> dict[str, str]:
    if not cookie_header:
        return {}
    c = http.cookies.SimpleCookie()
    try:
        c.load(cookie_header)
    except Exception:
        pass
    return {k: v.value for k, v in c.items()}


# ── fake session object for FlowEvaluator ────────────────────────────────────

class _FakeSession:
    """Minimal Session interface that FlowEvaluator expects."""
    def __init__(self, session_data: dict):
        self.cookies = session_data["cookies"]
        self.paths = session_data["paths"]


# ── preview server handler factory ───────────────────────────────────────────

def make_handler(bundle, evaluator: FlowEvaluator | None):
    artifact_map = {a.path: a for a in bundle.artifacts}
    index_page = "/index.html"

    class Handler(BaseHTTPRequestHandler):

        def _serve(self, method: str):
            bare_path = self.path.split("?")[0]
            raw_cookies = self.headers.get("Cookie")
            request_cookies = _parse_cookies(raw_cookies)

            # resolve or create smoketest session
            sid = request_cookies.get("smoketest_sid")
            sid, sess_data = _get_or_create_session(sid)

            # merge any cookies the browser already carries into the session
            sess_data["cookies"].update(
                {k: v for k, v in request_cookies.items() if k != "smoketest_sid"}
            )

            # record this request in history (before evaluation, same as
            # session_manager.add_or_update_session in production)
            sess_data["paths"].append({"path": bare_path, "method": method, "timestamp": time.time()})

            fake_session = _FakeSession(sess_data)

            # ── flow evaluation ───────────────────────────────────────────
            flow_result: FlowMatchResult | None = None
            if evaluator is not None:
                flow_result = evaluator.evaluate(fake_session, bare_path, {"method": method})
                if flow_result.matched:
                    print(
                        f"  [FLOW] {method} {bare_path} → "
                        + (f"302 {flow_result.redirect_to}" if flow_result.redirect_to
                           else f"rewrite → {flow_result.artifact_path}")
                    )

            # ── determine which artifact to serve ─────────────────────────
            lookup_path = bare_path
            status_code = 200

            if flow_result and flow_result.matched:
                status_code = flow_result.status_code
                if flow_result.redirect_to:
                    # synthetic redirect — no artifact needed
                    self._send_redirect(
                        flow_result.redirect_to,
                        sid,
                        flow_result.set_cookie,
                        flow_result.clear_cookie,
                        sess_data,
                    )
                    return
                if flow_result.artifact_path:
                    lookup_path = flow_result.artifact_path

            # root → primary path
            if lookup_path == "/":
                lookup_path = bundle.primary_path

            artifact = artifact_map.get(lookup_path)

            # baseline index page fallback
            if artifact is None and lookup_path == index_page:
                self._send_redirect(bundle.primary_path, sid, {}, [], sess_data)
                return

            if artifact is None:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self._set_sid_cookie(sid)
                self.end_headers()
                self.wfile.write(
                    f"404 Not found: {lookup_path}\n\nBundle paths:\n".encode()
                    + "\n".join(sorted(artifact_map)).encode()
                )
                return

            # ── normal artifact response ───────────────────────────────────
            self.send_response(status_code)

            content_type = None
            for header in artifact.headers:
                if isinstance(header, dict):
                    for key, value in header.items():
                        if key.lower() == "content-type":
                            content_type = value
                        self.send_header(key, value)
            if content_type is None:
                self.send_header("Content-Type", "application/octet-stream")

            # apply flow cookies to both session store and response headers
            if flow_result and flow_result.matched:
                for name, value in flow_result.set_cookie.items():
                    sess_data["cookies"][name] = value
                    self.send_header("Set-Cookie", f"{name}={value}; Path=/; HttpOnly")
                for name in flow_result.clear_cookie:
                    sess_data["cookies"].pop(name, None)
                    self.send_header(
                        "Set-Cookie", f"{name}=; Path=/; Max-Age=0; HttpOnly"
                    )

            self._set_sid_cookie(sid)
            self.end_headers()
            self.wfile.write(artifact.body_bytes)

        def _send_redirect(
            self,
            location: str,
            sid: str,
            set_cookie: dict,
            clear_cookie: list,
            sess_data: dict,
        ):
            self.send_response(302)
            self.send_header("Location", location)
            for name, value in set_cookie.items():
                sess_data["cookies"][name] = value
                self.send_header("Set-Cookie", f"{name}={value}; Path=/; HttpOnly")
            for name in clear_cookie:
                sess_data["cookies"].pop(name, None)
                self.send_header(
                    "Set-Cookie", f"{name}=; Path=/; Max-Age=0; HttpOnly"
                )
            self._set_sid_cookie(sid)
            self.end_headers()

        def _set_sid_cookie(self, sid: str):
            self.send_header("Set-Cookie", f"smoketest_sid={sid}; Path=/")

        def do_GET(self):
            self._serve("GET")

        def do_POST(self):
            # consume body so the connection stays clean
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            self._serve("POST")

        def log_message(self, fmt, *args):
            print(f"  {self.address_string()} {fmt % args}")

    return Handler


# ── main ──────────────────────────────────────────────────────────────────────

async def generate_bundle_for_path(
    path: str,
    cfg: GeneratorRuntimeConfig,
    *,
    verbose: bool = True,
):
    generator_cls = TracingGenerator if verbose else AgenticBundleGenerator
    generator = generator_cls(runtime_config=cfg)
    if verbose:
        print(f"\n=== GENERATING V2 BUNDLE: {path} ===")
    return await generator.generate_bundle(
        host="example.com",
        path=path,
        site_profile={"index_page": "/index.html"},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and preview or save a V2 smoketest bundle.")
    parser.add_argument("endpoint", nargs="?", default="/wp-login.php")
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--save-root", default=None, help="Save bundle as a preview.py-compatible snare root and exit.")
    parser.add_argument("--page-url", default="example.com")
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument("--quiet", action="store_true", help="Disable tracing output.")
    return parser.parse_args()


async def main():
    args = parse_args()
    path = args.endpoint
    cfg = build_smoketest_config(args.model_name)
    bundle = await generate_bundle_for_path(path, cfg, verbose=not args.quiet)

    print("\n=== FINAL BUNDLE ===")
    print("primary_path :", bundle.primary_path)
    print("used_fallback:", bundle.used_fallback)
    print("review_summary:", bundle.review_summary)
    print(f"artifacts ({len(bundle.artifacts)}):")
    for a in sorted(bundle.artifacts, key=lambda x: x.path):
        tag = "  [flow variant]" if a.path.startswith("/_flow/") else ""
        print(f"  {a.path:50s}  {a.kind:20s}  {len(a.body_bytes):>7,} bytes{tag}")

    evaluator: FlowEvaluator | None = None
    if bundle.flow_descriptor is not None:
        print(f"\n=== FLOW DESCRIPTOR ===")
        print_flow_descriptor(bundle.flow_descriptor)
        evaluator = FlowEvaluator()
        evaluator.register(bundle.primary_path, bundle.flow_descriptor)
    else:
        print("\n=== FLOW DESCRIPTOR: none generated ===")
        print("  (design node did not produce /_flow/ variants or dynamic_candidate paths)")
        print("  Try a login/admin path: smoketest_v2.py /wp-admin/login.php")

    if args.save_root:
        saved_root = save_bundle_to_snare_root(bundle, args.save_root, page_url=args.page_url)
        print("\n=== BUNDLE SAVED ===")
        print(f"snare_root: {saved_root}")
        print(f"preview:    python preview.py --snare-root {saved_root} --endpoint {bundle.primary_path}")
        return

    handler_class = make_handler(bundle, evaluator)
    server = ThreadingHTTPServer(("127.0.0.1", args.preview_port), handler_class)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("\n=== PREVIEW SERVER ===")
    print("SSH port forward (run on your LOCAL machine):")
    print(f"  ssh -L {args.preview_port}:127.0.0.1:{args.preview_port} proxmox-vm")
    print(f"\nThen open:  http://127.0.0.1:{args.preview_port}{bundle.primary_path}")
    print()
    if evaluator:
        print("Flow rules are live.  Cookies and request history persist across")
        print("requests in the same browser session.  To reset session state,")
        print("clear cookies or open a private window.")
    print("\nPress Ctrl+C to stop.")

    try:
        await asyncio.Event().wait()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPreview stopped.")