from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from tanner import __version__ as tanner_version


@dataclass
class PolicyDecision:
    """Data class to return from decide()."""
    override: bool
    detection: Optional[Dict[str, Any]] = None
    reason: str = ""
    lebo: str
    allowed_next: Optional[List[str]] = None  


class PolicyEngine:
    """
    Integrates the policy derived from telescope data into Tanner's decision logic.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_path: Optional[str] = None,
        mass_cutoff: float = 0.90,
        top_k: Optional[int] = None,
        default_status: int = 404,
        allow_first_request: bool = True,
        fail_open: bool = True,
        strip_query_string: bool = True,
    ):
        self.enabled = enabled
        self.mass_cutoff = mass_cutoff
        self.top_k = top_k
        self.default_status = default_status
        self.allow_first_request = allow_first_request
        self.fail_open = fail_open
        self.strip_query_string = strip_query_string

        self.logger = logging.getLogger("tanner.policy.PolicyEngine")

        # prev_path -> dict(next_path -> prob)
        self.transitions: Dict[str, Dict[str, float]] = {}

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load transitions JSON from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        with open(model_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        transitions = raw.get("transitions", raw)

        parsed: Dict[str, Dict[str, float]] = {}
        for prev, nxts in transitions.items():
            if not isinstance(nxts, dict):
                continue
            prev_n = self._normalize_path(prev)
            parsed[prev_n] = {self._normalize_path(n): float(p) for n, p in nxts.items()}

        self.transitions = parsed
        self.logger.info("Loaded policy model with %d states from %s", len(self.transitions), model_path)

    def decide(self, *, session: Any, path: str, data: Dict[str, Any]) -> PolicyDecision:
        """
        Decision method called in TannerServer.handle_event().
        path: request path (may include query string)
        session: Tanner session object
        data: raw event data from Snare (not yet used)
        """
        if not self.enabled:
            return PolicyDecision(override=False, reason="policy_disabled")

        cur = self._normalize_path(path)

        # Pull ordered path history from session. Because we don't know your Session implementation
        # details here, we try a few common shapes.
        history = self._extract_session_paths(session)

        # IMPORTANT: Session manager may have already appended the *current* path.
        # We want "previous state" excluding current request, so we drop a trailing current path.
        history_wo_current = history[:]
        if history_wo_current and history_wo_current[-1] == cur:
            history_wo_current = history_wo_current[:-1]

        if self.allow_first_request and len(history_wo_current) == 0:
            return PolicyDecision(override=False, reason="first_request_allowed")

        prev = history_wo_current[-1] if history_wo_current else None
        if not prev:
            return self._maybe_fail_open("no_prev_state")

        nxt_dist = self.transitions.get(prev)
        if not nxt_dist:
            return self._maybe_fail_open(f"no_model_for_prev={prev}")

        allowed_next = self._select_allowed(nxt_dist)

        if cur not in allowed_next:
            det = self._detection_status(self.default_status, name="policy_404")
            return PolicyDecision(
                override=True,
                detection=det,
                reason=f"path_not_allowed prev={prev} cur={cur}",
                allowed_next=allowed_next,
            )

        return PolicyDecision(override=False, reason=f"allowed prev={prev} cur={cur}", allowed_next=allowed_next)


    def _maybe_fail_open(self, reason: str) -> PolicyDecision:
        if self.fail_open:
            return PolicyDecision(override=False, reason=f"fail_open:{reason}")
        det = self._detection_status(self.default_status, name="policy_fail_closed")
        return PolicyDecision(override=True, detection=det, reason=f"fail_closed:{reason}")

    def _select_allowed(self, nxt_dist: Dict[str, float]) -> List[str]:
        """
        Pick allowed next endpoints by probability mass cutoff and/or top_k.
        """
        items = sorted(nxt_dist.items(), key=lambda kv: kv[1], reverse=True)
        allowed: List[str] = []

        total = 0.0
        for nxt, p in items:
            allowed.append(nxt)
            total += p
            if self.top_k is not None and len(allowed) >= self.top_k:
                break
            if self.mass_cutoff is not None and total >= self.mass_cutoff:
                break

        return allowed

    def _normalize_path(self, path: str) -> str:
        """
        Normalize to reduce key explosion:
        - collapse multiple slashes
        - optionally strip query string
        - remove trailing slash (except '/')
        """
        if not path:
            return "/"

        # collapse multiple slashes
        path = re.sub(r"/+", "/", path)

        if self.strip_query_string:
            path = path.split("?", 1)[0]

        if path != "/" and path.endswith("/"):
            path = path[:-1]

        return path

    def _extract_session_paths(self, session: Any) -> List[str]:
        """
        Try to extract an ordered list of request paths from the Tanner session.
        This tries a few patterns commonly used in Tanner-like code.
        """
        # 1) method accessor
        for meth in ("get_paths", "paths", "get_history", "history"):
            if hasattr(session, meth) and callable(getattr(session, meth)):
                try:
                    v = getattr(session, meth)()
                    return self._coerce_paths(v)
                except Exception:
                    pass

        # 2) attribute
        for attr in ("paths", "history", "requests"):
            if hasattr(session, attr):
                try:
                    v = getattr(session, attr)
                    return self._coerce_paths(v)
                except Exception:
                    pass

        return []

    def _coerce_paths(self, v: Any) -> List[str]:
        """
        Convert various shapes into a list[str], normalized.
        Supported shapes:
        - ["path", "path2", ...]
        - [{"path": "/x", ...}, ...]
        - [("/x", ts), ...] or [(ts, "/x"), ...]
        """
        paths: List[str] = []

        if v is None:
            return paths

        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    paths.append(self._normalize_path(item))
                elif isinstance(item, dict) and "path" in item:
                    paths.append(self._normalize_path(str(item["path"])))
                elif isinstance(item, tuple) and len(item) >= 2:
                    # attempt to find the string member
                    a, b = item[0], item[1]
                    if isinstance(a, str):
                        paths.append(self._normalize_path(a))
                    elif isinstance(b, str):
                        paths.append(self._normalize_path(b))
        return paths

    def _detection_status(self, status_code: int, name: str = "policy_status") -> Dict[str, Any]:
        """Build a Snare-compatible 'type 3' detection."""
        return {
            "name": name,
            "order": 0,
            "type": 3,
            "version": tanner_version,
            "payload": {"status_code": int(status_code)},
        }
