#!/usr/bin/env python3
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


def _normalize_path(path: str) -> str:
    normalized = (path or "").split("?", 1)[0]
    normalized = unquote(normalized)
    if not normalized:
        normalized = "/"
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized


def _build_candidates(request_path: str, index_page: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        if not path:
            return
        normalized = _normalize_path(path)
        if normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)

    add(request_path)

    normalized = _normalize_path(request_path)
    if normalized == "/":
        add(index_page)
    elif normalized.endswith("/"):
        trimmed = normalized[:-1] or "/"
        add(trimmed)
        if trimmed != "/":
            add(trimmed + index_page)
    else:
        add(normalized + "/")
        add(normalized + index_page)

    return candidates


def _extract_hash(entry):
    if isinstance(entry, dict):
        hash_name = entry.get("hash")
        return hash_name if isinstance(hash_name, str) and hash_name else None
    if isinstance(entry, str):
        return entry.lstrip("/")
    return None


def _extract_headers(entry, fallback_content_type: str | None = None) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []

    if isinstance(entry, dict):
        raw_headers = entry.get("headers", [])
        if isinstance(raw_headers, list):
            for header in raw_headers:
                if isinstance(header, dict):
                    for key, value in header.items():
                        if isinstance(key, str) and isinstance(value, str):
                            headers.append((key, value))

        content_type = entry.get("content_type")
        if isinstance(content_type, str) and content_type:
            headers = [(k, v) for (k, v) in headers if k.lower() != "content-type"]
            headers.append(("Content-Type", content_type))

    if fallback_content_type and not any(k.lower() == "content-type" for (k, _) in headers):
        headers.append(("Content-Type", fallback_content_type))

    return headers


def _guess_content_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise ValueError(f"meta.json must contain a JSON object at {meta_path}")
    return meta


def _resolve_entry(meta: dict, request_path: str, index_page: str):
    for candidate in _build_candidates(request_path, index_page):
        entry = meta.get(candidate)
        if entry is not None:
            return candidate, entry
    return None, None


def build_handler(meta: dict, page_dir: Path, preview_endpoint: str, index_page: str):
    class Handler(BaseHTTPRequestHandler):
        def _serve_from_entry(self, resolved_path: str, entry, status_code: int = 200) -> None:
            hash_name = _extract_hash(entry)
            if not hash_name:
                self.send_error(500, f"Invalid meta entry for {resolved_path}: missing hash")
                return

            file_path = page_dir / hash_name
            if not file_path.is_file():
                self.send_error(500, f"Missing cached file for {resolved_path}: {hash_name}")
                return

            body = file_path.read_bytes()
            headers = _extract_headers(entry, fallback_content_type=_guess_content_type(resolved_path))

            self.send_response(status_code)
            for key, value in headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()

            if self.command != "HEAD":
                self.wfile.write(body)

        def _serve_404(self) -> None:
            status_path, status_entry = _resolve_entry(meta, "/status_404", index_page)
            if status_entry is not None and status_path is not None:
                self._serve_from_entry(status_path, status_entry, status_code=404)
                return

            body = b"Not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):
            self._handle()

        def do_GET(self):
            self._handle()

        def _handle(self):
            request_path = _normalize_path(self.path)
            if request_path == "/":
                request_path = preview_endpoint

            resolved_path, entry = _resolve_entry(meta, request_path, index_page)
            if entry is None or resolved_path is None:
                self._serve_404()
                return

            self._serve_from_entry(resolved_path, entry)

        def log_message(self, fmt, *args):
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview cached Snare pages from meta.json/hash files.")
    parser.add_argument("--endpoint", required=True, help="Endpoint to preview, e.g. /setup.cgi")
    parser.add_argument(
        "--snare-root",
        default="/home/kaan/snare-data-prewarm-v1",
        help="Snare root containing pages/<page-url>/meta.json",
    )
    parser.add_argument("--page-url", default="example.com", help="Page directory under pages/")
    parser.add_argument("--index-page", default="/index.html", help="Index page for variant resolution")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    args = parser.parse_args()

    index_page = _normalize_path(args.index_page)
    endpoint = _normalize_path(args.endpoint)

    snare_root = Path(args.snare_root).resolve()
    page_dir = snare_root / "pages" / args.page_url
    meta_path = page_dir / "meta.json"

    meta = _load_meta(meta_path)
    resolved_path, entry = _resolve_entry(meta, endpoint, index_page)
    if entry is None or resolved_path is None:
        raise SystemExit(f"Endpoint not found in meta.json: {endpoint}")

    handler = build_handler(meta=meta, page_dir=page_dir, preview_endpoint=resolved_path, index_page=index_page)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print("=== CACHE PREVIEW SERVER ===")
    print(f"snare_root: {snare_root}")
    print(f"page_dir:   {page_dir}")
    print(f"endpoint:   {resolved_path}")
    print(f"url:        http://{args.host}:{args.port}{resolved_path}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
