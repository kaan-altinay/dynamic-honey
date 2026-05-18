"""Flow rule evaluator for V2 scripted interaction flows."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tanner.generator.agentic.models import FlowDescriptor, FlowRule


@dataclass
class FlowMatchResult:
    """Outcome of evaluating flow rules against a request + session."""
    matched: bool
    artifact_path: str | None = None     # rewrite: serve this /_flow/ meta key
    redirect_to: str | None = None       # synthetic redirect
    status_code: int = 200
    set_cookie: dict[str, str] = field(default_factory=dict)
    clear_cookie: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)


class FlowEvaluator:
    """
    Evaluates scripted flow rules against live session state.

    One FlowEvaluator instance lives on TannerServer.  Flow descriptors are
    registered as bundles are generated (dynamic V2) or loaded on startup
    (cache-only V2 via prewarm_cache_v2).  All rules from all registered
    descriptors compete on every request; the highest-priority matching rule
    wins.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        # primary_path -> FlowDescriptor
        self._flows: dict[str, FlowDescriptor] = {}

    # ── registration ──────────────────────────────────────────────────────────

    def register(self, key: str, descriptor: FlowDescriptor) -> None:
        """Register a flow descriptor keyed by bundle primary_path."""
        self._flows[key] = descriptor
        self.logger.info(
            "FlowEvaluator: registered %d rule(s) for key %r",
            len(descriptor.rules),
            key,
        )

    def load_from_dict(self, key: str, raw: dict) -> None:
        """Parse and register a FlowDescriptor from a raw dict (e.g. from JSON)."""
        try:
            descriptor = FlowDescriptor.model_validate(raw)
            self.register(key, descriptor)
        except Exception as exc:
            self.logger.warning(
                "FlowEvaluator: failed to load descriptor for key %r: %s", key, exc
            )

    def clear(self) -> None:
        """Remove all registered descriptors (used in tests)."""
        self._flows.clear()

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, session: Any, path: str, data: dict) -> FlowMatchResult:
        """
        Evaluate all registered flow rules against the current request and
        session state.  Returns the first match in priority-descending order,
        or FlowMatchResult(matched=False) if no rule fires.

        Args:
            session: Live Tanner Session object (has .paths, .cookies)
            path:    Requested path, query-string stripped
            data:    Full Tanner event data dict (has "method" key)
        """
        method = (data.get("method") or "GET").upper()

        all_rules: list[FlowRule] = []
        for descriptor in self._flows.values():
            all_rules.extend(descriptor.rules)

        # Highest priority first; stable sort preserves insertion order on ties
        all_rules.sort(key=lambda r: r.priority, reverse=True)

        for rule in all_rules:
            if self._rule_matches(rule, path, method, session):
                return self._build_result(rule)

        return FlowMatchResult(matched=False)

    # ── internals ─────────────────────────────────────────────────────────────

    def _rule_matches(
        self, rule: FlowRule, path: str, method: str, session: Any
    ) -> bool:
        if rule.match_path != path:
            return False

        cond = rule.condition
        if cond is None:
            return True

        # HTTP method
        if cond.method is not None and cond.method.upper() != method:
            return False

        # Cookie checks
        if cond.requires_cookie is not None:
            if cond.requires_cookie not in session.cookies:
                return False
        if cond.missing_cookie is not None:
            if cond.missing_cookie in session.cookies:
                return False

        # Previous path check (exclude current request already appended by session_manager)
        if cond.requires_prev_path is not None:
            history = [e["path"].split("?")[0] for e in session.paths]
            prior = history[:-1] if history else []
            if not prior or prior[-1] != cond.requires_prev_path:
                return False

        current_ts, prior_posts = self._post_history(session, path)

        # POST count threshold (legacy inclusive count, including current request when present)
        if cond.min_post_count_to_path is not None:
            post_count = len(prior_posts)
            if method == "POST":
                post_count += 1
            if post_count < cond.min_post_count_to_path:
                return False

        if cond.min_prior_post_count_to_path is not None:
            if len(prior_posts) < cond.min_prior_post_count_to_path:
                return False

        if cond.lockout_active is not None:
            threshold = cond.min_prior_post_count_to_path or cond.min_post_count_to_path
            window_seconds = cond.lockout_window_seconds
            if threshold is None or window_seconds is None:
                return False
            active = self._lockout_active(prior_posts, current_ts, threshold, window_seconds)
            if active != cond.lockout_active:
                return False

        return True

    @staticmethod
    def _post_history(session: Any, path: str) -> tuple[float, list[float]]:
        history = list(getattr(session, "paths", []) or [])
        current_ts = time.time()
        prior_events = history
        if history:
            last = history[-1]
            current_ts = float(last.get("timestamp", current_ts))
            if last.get("path", "").split("?")[0] == path:
                prior_events = history[:-1]
        prior_posts = [
            float(e.get("timestamp", current_ts))
            for e in prior_events
            if e.get("path", "").split("?")[0] == path
            and e.get("method", "GET").upper() == "POST"
        ]
        return current_ts, prior_posts

    @staticmethod
    def _lockout_active(prior_posts: list[float], current_ts: float, threshold: int, window_seconds: int) -> bool:
        attempts = 0
        lockout_until: float | None = None
        for ts in prior_posts:
            if lockout_until is not None and ts < lockout_until:
                continue
            if lockout_until is not None and ts >= lockout_until:
                attempts = 0
                lockout_until = None
            attempts += 1
            if attempts >= threshold:
                lockout_until = ts + window_seconds
                attempts = 0
        return lockout_until is not None and current_ts < lockout_until

    @staticmethod
    def _build_result(rule: FlowRule) -> FlowMatchResult:
        resp = rule.response
        flat_headers: dict[str, str] = {}
        for hdr in resp.headers:
            flat_headers.update(hdr)
        return FlowMatchResult(
            matched=True,
            artifact_path=resp.artifact_path,
            redirect_to=resp.redirect_to,
            status_code=resp.status_code,
            set_cookie=dict(resp.set_cookie),
            clear_cookie=list(resp.clear_cookie),
            headers=flat_headers,
        )
