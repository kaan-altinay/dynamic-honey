from __future__ import annotations

import html
import re
import shlex
import subprocess
import tempfile
from typing import Iterable
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from tanner.generator.agentic.models import CommandResult, GeneratorRuntimeConfig, ResearchResult


_SEARCH_URL = "https://duckduckgo.com/html/?q={query}"
_SAFE_COMMANDS = {"cat", "grep", "head", "tail", "wc", "printf", "echo", "xmllint", "tidy"}
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_RESULT_LINK_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<label>.*?)</a>', re.I)
_SNIPPET_RE = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(?P<div_snippet>.*?)</div>', re.I)


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
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ResearchResult(query=url)

    request = Request(url, headers={"User-Agent": "TannerAgenticGenerator/1.0"})
    try:
        with urlopen(request, timeout=runtime_config.role_config("design").timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception:
        return ResearchResult(query=url)

    text = _strip_markup(body)
    return ResearchResult(
        query=url,
        snippets=_truncate_lines([text], runtime_config.max_tool_response_chars),
        references=[url],
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
