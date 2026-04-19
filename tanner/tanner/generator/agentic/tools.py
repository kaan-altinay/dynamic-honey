from __future__ import annotations

import html
import mimetypes
import re
import shlex
import subprocess
import tempfile
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

from tanner.generator.agentic.models import (
    AssetCandidate,
    AssetFetchKind,
    CommandResult,
    FetchedAsset,
    GeneratorRuntimeConfig,
    ReferencePage,
    ResearchResult,
)


_SEARCH_URL = "https://duckduckgo.com/html/?q={query}"
_SAFE_COMMANDS = {"cat", "grep", "head", "tail", "wc", "printf", "echo", "xmllint", "tidy"}
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_RESULT_LINK_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<label>.*?)</a>', re.I)
_SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(?P<div_snippet>.*?)</div>', re.I)
_CSS_URL_RE = re.compile(r"url\((?P<quote>['\"]?)(?P<url>[^)'\"]+)(?P=quote)\)", re.I)
_MAX_REFERENCE_HTML_BYTES = 350_000
_MAX_FETCHED_ASSET_BYTES = 1_000_000
_ALLOWED_FETCH_CONTENT_TYPES = (
    "text/html",
    "text/css",
    "application/javascript",
    "text/javascript",
    "application/x-javascript",
    "image/",
)


def _strip_markup(value: str) -> str:
    without_tags = _TAG_RE.sub(" ", value)
    normalized = _WHITESPACE_RE.sub(" ", html.unescape(without_tags)).strip()
    return normalized


def _truncate_lines(values: Iterable[str], limit: int) -> list[str]:
    output = []
    remaining = limit
    for value in values:
        if not value:
            continue
        if remaining <= 0:
            break
        chunk = value[:remaining].strip()
        if not chunk:
            continue
        output.append(chunk)
        remaining -= len(chunk)
    return output


def _guess_asset_kind(url: str, tag: str, rel: str = "") -> AssetFetchKind:
    lowered_url = url.lower()
    rel = rel.lower()
    if tag == "link" and "stylesheet" in rel:
        return "stylesheet"
    if tag == "script":
        return "script"
    if tag == "link" and "icon" in rel:
        return "icon"
    if any(lowered_url.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"]):
        return "image" if not lowered_url.endswith(".ico") else "icon"
    if any(lowered_url.endswith(ext) for ext in [".woff", ".woff2", ".ttf", ".otf"]):
        return "font"
    return "other"


def _sanitize_path_segment(value: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return value or fallback


def _default_local_path(url: str, kind: AssetFetchKind) -> str:
    parsed = urlparse(url)
    basename = parsed.path.rstrip("/").split("/")[-1] or "asset"
    basename = _sanitize_path_segment(basename, "asset")
    if "." not in basename:
        default_ext = {
            "image": ".png",
            "icon": ".ico",
            "stylesheet": ".css",
            "script": ".js",
            "font": ".woff2",
            "other": ".bin",
        }[kind]
        basename = basename + default_ext
    return "/assets/{}".format(basename)


class _ReferenceHtmlParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.in_title = False
        self.title_parts: list[str] = []
        self.asset_candidates: list[AssetCandidate] = []

    def handle_starttag(self, tag: str, attrs):
        attr_map = {key.lower(): value for key, value in attrs if isinstance(key, str) and isinstance(value, str)}
        if tag == "title":
            self.in_title = True
            return
        if tag == "img":
            src = attr_map.get("src")
            if src:
                resolved = urljoin(self.base_url, src)
                self.asset_candidates.append(
                    AssetCandidate(
                        source_url=resolved,
                        kind=_guess_asset_kind(resolved, tag),
                        tag=tag,
                        local_path_hint=_default_local_path(resolved, "image"),
                        note=attr_map.get("alt", "image asset"),
                    )
                )
            return
        if tag == "script":
            src = attr_map.get("src")
            if src:
                resolved = urljoin(self.base_url, src)
                self.asset_candidates.append(
                    AssetCandidate(
                        source_url=resolved,
                        kind="script",
                        tag=tag,
                        local_path_hint=_default_local_path(resolved, "script"),
                        note="script asset",
                    )
                )
            return
        if tag == "link":
            href = attr_map.get("href")
            if not href:
                return
            rel = attr_map.get("rel", "")
            kind = _guess_asset_kind(href, tag, rel=rel)
            if kind in {"stylesheet", "icon"}:
                resolved = urljoin(self.base_url, href)
                self.asset_candidates.append(
                    AssetCandidate(
                        source_url=resolved,
                        kind=kind,
                        tag=tag,
                        local_path_hint=_default_local_path(resolved, kind),
                        note=rel or "linked asset",
                    )
                )

    def handle_endtag(self, tag: str):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str):
        if self.in_title:
            self.title_parts.append(data)

    @property
    def title(self) -> str:
        return _WHITESPACE_RE.sub(" ", "".join(self.title_parts)).strip()


def _parse_reference_html(html_text: str, base_url: str) -> tuple[str, list[AssetCandidate]]:
    parser = _ReferenceHtmlParser(base_url)
    parser.feed(html_text)
    return parser.title, parser.asset_candidates

def extract_asset_candidates(html_text: str, base_url: str) -> list[AssetCandidate]:
    _, asset_candidates = _parse_reference_html(html_text, base_url)
    return asset_candidates


def extract_css_asset_candidates(css_text: str, base_url: str) -> list[AssetCandidate]:
    candidates = []
    for match in _CSS_URL_RE.finditer(css_text):
        candidate_url = match.group("url").strip()
        if not candidate_url or candidate_url.startswith(("data:", "#")):
            continue
        resolved = urljoin(base_url, candidate_url)
        kind = _guess_asset_kind(resolved, "style")
        candidates.append(
            AssetCandidate(
                source_url=resolved,
                kind=kind,
                tag="style",
                local_path_hint=_default_local_path(resolved, kind),
                note="css-referenced asset",
            )
        )
    return candidates


def _dedupe_asset_candidates(candidates: list[AssetCandidate]) -> list[AssetCandidate]:
    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate.source_url in seen:
            continue
        seen.add(candidate.source_url)
        deduped.append(candidate)
    return deduped


def _open_url(url: str, timeout: int):
    request = Request(url, headers={"User-Agent": "TannerAgenticGenerator/1.0"})
    return urlopen(request, timeout=timeout)


def web_research(query: str, runtime_config: GeneratorRuntimeConfig) -> ResearchResult:
    safe_query = query.strip()[:200]
    if not safe_query:
        return ResearchResult(query=query)

    request = Request(
        _SEARCH_URL.format(query=quote(safe_query)),
        headers={"User-Agent": "TannerAgenticGenerator/1.0"},
    )
    try:
        with urlopen(request, timeout=runtime_config.role_config("expert").timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception:
        return ResearchResult(query=safe_query)

    references = []
    for match in _RESULT_LINK_RE.finditer(body):
        href = html.unescape(match.group("href"))
        if href.startswith("http"):
            references.append(href)
        if len(references) >= 3:
            break

    snippet_values = []
    for match in _SNIPPET_RE.finditer(body):
        snippet = match.group("snippet") or match.group("div_snippet") or ""
        snippet_values.append(_strip_markup(snippet))
        if len(snippet_values) >= 3:
            break

    return ResearchResult(
        query=safe_query,
        snippets=_truncate_lines(snippet_values, runtime_config.max_tool_response_chars),
        references=references,
    )


def inspect_reference(url: str, runtime_config: GeneratorRuntimeConfig) -> ResearchResult:
    reference_page = fetch_reference_page(url, runtime_config)
    return ResearchResult(
        query=url,
        snippets=_truncate_lines([reference_page.text_excerpt], runtime_config.max_tool_response_chars),
        references=[reference_page.final_url],
    )


def fetch_reference_page(url: str, runtime_config: GeneratorRuntimeConfig) -> ReferencePage:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ReferencePage(url=url, final_url=url)

    try:
        with _open_url(url, runtime_config.role_config("design").timeout) as response:
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            body = response.read(_MAX_REFERENCE_HTML_BYTES).decode("utf-8", errors="replace")
    except Exception:
        return ReferencePage(url=url, final_url=url)

    if "html" not in content_type.lower() and "xml" in content_type.lower():
        return ReferencePage(url=url, final_url=final_url)

    title, asset_candidates = _parse_reference_html(body, final_url)

    stylesheet_candidates = [candidate for candidate in asset_candidates if candidate.kind == "stylesheet"]
    for stylesheet_candidate in stylesheet_candidates[:3]:
        try:
            with _open_url(stylesheet_candidate.source_url, runtime_config.role_config("design").timeout) as response:
                css_text = response.read(200_000).decode("utf-8", errors="replace")
        except Exception:
            continue
        asset_candidates.extend(extract_css_asset_candidates(css_text, stylesheet_candidate.source_url))

    return ReferencePage(
        url=url,
        final_url=final_url,
        title=title,
        text_excerpt=_strip_markup(body)[: runtime_config.max_tool_response_chars],
        asset_candidates=_dedupe_asset_candidates(asset_candidates),
    )


def fetch_static_asset(
    source_url: str,
    local_path: str,
    kind: AssetFetchKind,
    runtime_config: GeneratorRuntimeConfig,
    asset_id: str,
    required_for_artifact_ids: list[str] | None = None,
) -> FetchedAsset:
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("unsupported asset URL scheme")

    with _open_url(source_url, runtime_config.role_config("design").timeout) as response:
        content_type = response.headers.get("Content-Type") or mimetypes.guess_type(source_url)[0] or "application/octet-stream"
        lowered_content_type = content_type.lower()
        if not any(
            lowered_content_type == allowed or lowered_content_type.startswith(allowed)
            for allowed in _ALLOWED_FETCH_CONTENT_TYPES
        ):
            raise ValueError("unsupported fetched asset content-type {}".format(content_type))
        body = response.read(_MAX_FETCHED_ASSET_BYTES + 1)

    if len(body) > _MAX_FETCHED_ASSET_BYTES:
        raise ValueError("fetched asset exceeds size limit")

    return FetchedAsset(
        asset_id=asset_id,
        source_url=source_url,
        local_path=local_path,
        kind=kind,
        content_type=content_type,
        body_bytes=body,
        required_for_artifact_ids=required_for_artifact_ids or [],
    )


def sandbox_command(command: str, runtime_config: GeneratorRuntimeConfig) -> CommandResult:
    command = (command or "").strip()
    if not command:
        return CommandResult(command=command, exit_code=1, stderr="empty command")

    try:
        parts = shlex.split(command)
    except ValueError as error:
        return CommandResult(command=command, exit_code=1, stderr=str(error))

    binary = parts[0]
    if binary not in _SAFE_COMMANDS:
        return CommandResult(command=command, exit_code=1, stderr="command is not allowlisted")

    for token in parts[1:]:
        if any(marker in token for marker in ["&&", "||", ";", "|", "`", "$(", ".."]):
            return CommandResult(command=command, exit_code=1, stderr="command contains disallowed token")
        if token.startswith("/"):
            return CommandResult(command=command, exit_code=1, stderr="absolute paths are not permitted")

    with tempfile.TemporaryDirectory(prefix="tanner-design-tool-") as temp_dir:
        try:
            completed = subprocess.run(
                parts,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=runtime_config.command_timeout,
                check=False,
            )
        except Exception as error:
            return CommandResult(command=command, exit_code=1, stderr=str(error))

    stdout = completed.stdout[: runtime_config.max_command_output_chars]
    stderr = completed.stderr[: runtime_config.max_command_output_chars]
    return CommandResult(command=command, exit_code=completed.returncode, stdout=stdout, stderr=stderr)
