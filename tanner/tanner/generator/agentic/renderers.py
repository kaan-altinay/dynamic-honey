from __future__ import annotations

import html
import re
from typing import Iterable

from tanner.generator.agentic.models import ArtifactDraft, GeneratedArtifact


def _normalize_headers(headers_hint: list[dict[str, str]], content_type: str) -> list[dict[str, str]]:
    normalized = []
    has_content_type = False
    for header in headers_hint:
        if not isinstance(header, dict):
            continue
        for key, value in header.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            normalized.append({key: value})
            if key.lower() == "content-type":
                has_content_type = True
    if not has_content_type:
        normalized.append({"Content-Type": content_type})
    return normalized


def _join_lines(values: Iterable[str]) -> str:
    return "\n".join(value for value in values if isinstance(value, str) and value)


def _render_html(draft: ArtifactDraft) -> bytes:
    model = draft.content_model
    title = html.escape(model.get("title", "Generated Resource"))
    heading = html.escape(model.get("heading", title))
    paragraphs = [html.escape(value) for value in model.get("paragraphs", []) if isinstance(value, str)]
    nav_links = [link for link in model.get("nav_links", []) if isinstance(link, dict)]
    stylesheet_links = [link for link in model.get("linked_stylesheets", []) if isinstance(link, str)]
    script_links = [link for link in model.get("linked_scripts", []) if isinstance(link, str)]
    form = model.get("form") if isinstance(model.get("form"), dict) else None
    footer = html.escape(model.get("footer", ""))

    head_parts = ["<meta charset=\"utf-8\">", "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">", "<title>{}</title>".format(title)]
    for stylesheet in stylesheet_links:
        head_parts.append('<link rel="stylesheet" href="{}">'.format(html.escape(stylesheet, quote=True)))

    nav_html = ""
    if nav_links:
        nav_items = [
            '<li><a href="{href}">{label}</a></li>'.format(
                href=html.escape(link.get("href", "/"), quote=True),
                label=html.escape(link.get("label", "Link")),
            )
            for link in nav_links
        ]
        nav_html = "<nav><ul>{}</ul></nav>".format("".join(nav_items))

    paragraph_html = "".join("<p>{}</p>".format(paragraph) for paragraph in paragraphs)
    form_html = ""
    if form:
        action = html.escape(form.get("action", draft.path), quote=True)
        method = html.escape(form.get("method", "post"), quote=True)
        fields = []
        for field in form.get("fields", []):
            if not isinstance(field, dict):
                continue
            name = html.escape(field.get("name", "field"), quote=True)
            label = html.escape(field.get("label", name))
            field_type = html.escape(field.get("type", "text"), quote=True)
            fields.append(
                '<label>{label}<input type="{type}" name="{name}" autocomplete="off"></label>'.format(
                    label=label,
                    type=field_type,
                    name=name,
                )
            )
        submit_label = html.escape(form.get("submit_label", "Continue"))
        form_html = '<form action="{action}" method="{method}">{fields}<button type="submit">{submit}</button></form>'.format(
            action=action,
            method=method,
            fields="".join(fields),
            submit=submit_label,
        )

    script_html = "".join(
        '<script src="{}"></script>'.format(html.escape(script, quote=True)) for script in script_links
    )

    document = "".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\"><head>",
            "".join(head_parts),
            "</head><body>",
            nav_html,
            "<main><section><h1>{}</h1>{}{}</section></main>".format(heading, paragraph_html, form_html),
            "<footer><small>{}</small></footer>".format(footer),
            script_html,
            "</body></html>",
        ]
    )
    return document.encode("utf-8")


def _normalize_comment_text(comment: str, fmt: str) -> str:
    if not isinstance(comment, str):
        return ""
    normalized = comment.strip()
    if fmt == "php":
        normalized = re.sub(r"^(?:/\/+\s*)+", "", normalized)
    else:
        normalized = re.sub(r"^(?:#+\s*)+", "", normalized)
    return normalized.strip()

def _render_config_text(draft: ArtifactDraft) -> bytes:
    model = draft.content_model
    fmt = model.get("format", "env")
    comment = model.get("comment")
    entries = [entry for entry in model.get("entries", []) if isinstance(entry, dict)]
    lines = []
    if isinstance(comment, str) and comment:
        prefix = "#" if fmt != "php" else "//"
        lines.append("{} {}".format(prefix, _normalize_comment_text(comment, fmt)))

    if fmt == "php":
        lines.append("<?php")
        for entry in entries:
            key = entry.get("key")
            value = entry.get("value")
            if isinstance(key, str) and isinstance(value, str):
                lines.append("define('{}', '{}');".format(key, value.replace("'", "\\'")))
    else:
        for entry in entries:
            key = entry.get("key")
            value = entry.get("value")
            if isinstance(key, str) and isinstance(value, str):
                lines.append("{}={}".format(key, value))

    return (_join_lines(lines) + "\n").encode("utf-8")


def _render_stylesheet(draft: ArtifactDraft) -> bytes:
    rules = []
    for rule in draft.content_model.get("rules", []):
        if not isinstance(rule, dict):
            continue
        selector = rule.get("selector")
        declarations = rule.get("declarations")
        if not isinstance(selector, str) or not isinstance(declarations, dict):
            continue
        declaration_lines = [
            "  {}: {};".format(property_name, property_value)
            for property_name, property_value in declarations.items()
            if isinstance(property_name, str) and isinstance(property_value, str)
        ]
        if declaration_lines:
            rules.append("{} {{\n{}\n}}".format(selector, "\n".join(declaration_lines)))
    return (_join_lines(rules) + "\n").encode("utf-8")


def _render_javascript(draft: ArtifactDraft) -> bytes:
    lines = [line for line in draft.content_model.get("lines", []) if isinstance(line, str)]
    return (_join_lines(lines) + "\n").encode("utf-8")


def _render_plaintext(draft: ArtifactDraft) -> bytes:
    lines = [line for line in draft.content_model.get("lines", []) if isinstance(line, str)]
    return (_join_lines(lines) + "\n").encode("utf-8")


def _render_sitemap(draft: ArtifactDraft) -> bytes:
    urls = [url for url in draft.content_model.get("urls", []) if isinstance(url, str)]
    body = "".join("<url><loc>{}</loc></url>".format(html.escape(url)) for url in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{}</urlset>'.format(body)).encode("utf-8")


def render_artifact(draft: ArtifactDraft) -> GeneratedArtifact:
    if draft.kind == "html_page":
        body = _render_html(draft)
        content_type = "text/html; charset=utf-8"
    elif draft.kind == "config_text":
        body = _render_config_text(draft)
        content_type = "text/plain; charset=utf-8"
    elif draft.kind == "stylesheet":
        body = _render_stylesheet(draft)
        content_type = "text/css; charset=utf-8"
    elif draft.kind == "javascript":
        body = _render_javascript(draft)
        content_type = "application/javascript; charset=utf-8"
    elif draft.kind in {"robots_txt", "credential_bait", "log_excerpt", "backup_manifest"}:
        body = _render_plaintext(draft)
        content_type = "text/plain; charset=utf-8"
    elif draft.kind == "sitemap_xml":
        body = _render_sitemap(draft)
        content_type = "application/xml; charset=utf-8"
    else:
        raise ValueError("Unsupported artifact kind {}".format(draft.kind))

    return GeneratedArtifact(
        path=draft.path,
        kind=draft.kind,
        headers=_normalize_headers(draft.headers_hint, content_type),
        body_bytes=body,
        status_code=200,
        source_artifact_id=draft.artifact_id,
        artifact_scope="static_file",
    )
