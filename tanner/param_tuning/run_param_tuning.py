#!/usr/bin/env python3
import argparse
import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASELINE = {
    "max_bundle_artifacts": 8,
    "max_bundle_bytes": 2_000_000,
    "expert_temperature": 0.0,
    "design_temperature": 0.0,
    "coder_temperature": 0.1,
    "review_temperature": 0.0,
    "design_max_tokens": 2800,
    "coder_max_tokens": 1600,
    "max_review_loops": 1,
    "max_design_validation_loops": 2,
}

DEFAULT_ENDPOINTS = [
    "/boaform/admin/formLogin",
    "/api/v1/pods",
    "/containers/json",
    "/.env",
]

_FAMILY_SLUG_AXES: dict[str, list[str]] = {
    "A": ["max_bundle_artifacts", "max_bundle_bytes"],
    "B": ["design_max_tokens", "coder_max_tokens"],
    "C": ["max_review_loops", "max_design_validation_loops"],
    "D": ["expert_temperature", "design_temperature", "coder_temperature", "review_temperature"],
}

_PROGRESS_FILE = "run_state.json"
_RUN_CONFIG_FILE = "run_config.json"
_PREVIEW_COMMANDS_FILE = "preview_commands.txt"
# Exit code returned to the matrix runner when a rate-limit error is
# detected in a bundle result.  Distinct from 1 (unexpected error) so
# the caller can tell the difference and stop cleanly.
_EXIT_RATE_LIMITED = 2
# Substring embedded by the expert-node heuristic-fallback handler when
# it catches a RateLimitError and downgrades to a heuristic spec.
_RATE_LIMIT_MARKER = "RateLimitError"


def _slugify_endpoint(endpoint: str) -> str:
    slug = endpoint.strip().lstrip("/") or "root"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug.strip("-") or "root"


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def _auto_params_slug(family: str, args) -> str:
    family = family.upper()
    if family == "BASE":
        changed_axes = [
            axis
            for axis, baseline_value in BASELINE.items()
            if getattr(args, axis, None) not in (None, baseline_value)
        ]
        return "params_baseline" if not changed_axes else "params_custom"

    axes = _FAMILY_SLUG_AXES.get(family, [])
    parts = []
    for axis in axes:
        value = getattr(args, axis, None)
        if value is not None:
            parts.append(_format_param_value(value))
    return "params_" + "_".join(parts) if parts else "params_default"


def _next_family_run_dir(run_root: Path, family: str) -> Path:
    family_dir = run_root / f"family_{family}"
    family_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for child in family_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"run(\d+)", child.name)
        if match:
            max_n = max(max_n, int(match.group(1)))
    run_dir = family_dir / f"run{max_n + 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _apply_overrides(cfg, args, params_dir: Path):
    updates = {
        "checkpoint_path": str(params_dir / "runtime" / "tanner-agentic-checkpoints.sqlite"),
        "review_log_path": str(params_dir / "runtime" / "tanner-agentic-review-log.json"),
    }
    if args.max_bundle_artifacts is not None:
        updates["max_bundle_artifacts"] = args.max_bundle_artifacts
    if args.max_bundle_bytes is not None:
        updates["max_bundle_bytes"] = args.max_bundle_bytes
    if args.max_review_loops is not None:
        updates["max_review_loops"] = args.max_review_loops
    if args.max_design_validation_loops is not None:
        updates["max_design_validation_loops"] = args.max_design_validation_loops
    cfg = cfg.model_copy(update=updates)

    roles = dict(cfg.roles)
    if args.expert_temperature is not None:
        roles["expert"] = roles["expert"].model_copy(update={"temperature": args.expert_temperature})
    if args.design_temperature is not None:
        roles["design"] = roles["design"].model_copy(update={"temperature": args.design_temperature})
    if args.coder_temperature is not None:
        roles["coder"] = roles["coder"].model_copy(update={"temperature": args.coder_temperature})
    if args.review_temperature is not None:
        roles["review"] = roles["review"].model_copy(update={"temperature": args.review_temperature})
    if args.design_max_tokens is not None:
        roles["design"] = roles["design"].model_copy(update={"max_tokens": args.design_max_tokens})
    if args.coder_max_tokens is not None:
        roles["coder"] = roles["coder"].model_copy(update={"max_tokens": args.coder_max_tokens})
    return cfg.model_copy(update={"roles": roles})


def _progress_path(params_dir: Path) -> Path:
    return params_dir / _PROGRESS_FILE


def _run_config_path(params_dir: Path) -> Path:
    return params_dir / _RUN_CONFIG_FILE


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_progress(params_dir: Path) -> dict | None:
    progress = _read_json(_progress_path(params_dir))
    if progress is not None:
        return progress

    run_config = _read_json(_run_config_path(params_dir))
    if run_config is None:
        return None

    return {
        "family": run_config.get("family"),
        "family_run_dir": run_config.get("family_run_dir"),
        "params_dir": run_config.get("run_dir") or str(params_dir),
        "params_slug": run_config.get("params_slug") or params_dir.name,
        "page_url": run_config.get("page_url"),
        "endpoints": run_config.get("endpoints", []),
        "runtime_config": run_config.get("runtime_config"),
        "results": run_config.get("results", []),
        "completed_endpoint_count": len(run_config.get("results", [])),
        "complete": True,
    }


def _validate_progress_results(results: list[dict], endpoints: list[str]) -> None:
    if len(results) > len(endpoints):
        raise SystemExit("run_state.json records more completed endpoints than expected")

    for idx, summary in enumerate(results):
        expected_endpoint = endpoints[idx]
        actual_endpoint = summary.get("endpoint")
        if actual_endpoint != expected_endpoint:
            raise SystemExit(
                "run_state.json endpoint order mismatch at index {}: expected {!r}, found {!r}".format(
                    idx,
                    expected_endpoint,
                    actual_endpoint,
                )
            )


def _resume_results_and_pending_endpoints(
    params_dir: Path,
    endpoints: list[str],
) -> tuple[list[dict], list[str], bool]:
    progress = _load_progress(params_dir)
    if progress is None:
        return [], list(endpoints), False

    saved_endpoints = progress.get("endpoints")
    if saved_endpoints:
        endpoints = list(saved_endpoints)

    results = list(progress.get("results", []))
    _validate_progress_results(results, endpoints)
    completed_count = len(results)
    is_complete = bool(progress.get("complete")) and completed_count == len(endpoints)
    if is_complete:
        return results, [], True

    pending_endpoints = list(endpoints[completed_count:])
    for endpoint in pending_endpoints:
        endpoint_dir = params_dir / _slugify_endpoint(endpoint)
        if endpoint_dir.exists():
            shutil.rmtree(endpoint_dir)
    return results, pending_endpoints, False


def _write_progress(
    params_dir: Path,
    *,
    family: str,
    family_run_dir: Path,
    params_slug: str,
    page_url: str,
    endpoints: list[str],
    cfg,
    results: list[dict],
    complete: bool,
) -> None:
    payload = {
        "family": family,
        "family_run_dir": str(family_run_dir),
        "params_dir": str(params_dir),
        "params_slug": params_slug,
        "page_url": page_url,
        "endpoints": endpoints,
        "runtime_config": cfg.model_dump(mode="json"),
        "results": results,
        "completed_endpoint_count": len(results),
        "complete": complete,
    }
    _write_json(_progress_path(params_dir), payload)


def _write_run_outputs(
    params_dir: Path,
    *,
    family: str,
    family_run_dir: Path,
    params_slug: str,
    page_url: str,
    endpoints: list[str],
    cfg,
    summaries: list[dict],
) -> None:
    run_config = {
        "family": family,
        "family_run_dir": str(family_run_dir),
        "params_slug": params_slug,
        "page_url": page_url,
        "endpoints": endpoints,
        "runtime_config": cfg.model_dump(mode="json"),
        "results": summaries,
        "run_dir": str(params_dir),
    }
    _write_json(_run_config_path(params_dir), run_config)

    preview_lines = []
    for summary in summaries:
        preview_lines.append(
            "python preview.py --snare-root {root} --endpoint {endpoint}".format(
                root=params_dir / summary["folder"],
                endpoint=summary["primary_path"],
            )
        )
    (params_dir / _PREVIEW_COMMANDS_FILE).write_text("\n".join(preview_lines) + "\n")


def _load_smoketest_api():
    from smoketest_v2 import build_smoketest_config, generate_bundle_for_path, save_bundle_to_snare_root

    return build_smoketest_config, generate_bundle_for_path, save_bundle_to_snare_root


async def _run_endpoint(endpoint: str, cfg, params_dir: Path, page_url: str) -> dict:
    _, generate_bundle_for_path, save_bundle_to_snare_root = _load_smoketest_api()
    endpoint_dir = params_dir / _slugify_endpoint(endpoint)
    endpoint_dir.mkdir(parents=True, exist_ok=False)
    bundle = await generate_bundle_for_path(endpoint, cfg, verbose=False)
    save_bundle_to_snare_root(bundle, endpoint_dir, page_url=page_url)
    return {
        "endpoint": endpoint,
        "folder": endpoint_dir.name,
        "primary_path": bundle.primary_path,
        "artifact_count": len(bundle.artifacts),
        "used_fallback": bundle.used_fallback,
        "review_summary": bundle.review_summary,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run smoketest-style bundle generation across a stable endpoint panel "
            "and save the results under runs/family_<FAMILY>/run<N>/params_<slug>/ ."
        )
    )
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--page-url", default="example.com")
    parser.add_argument(
        "--device-endpoint",
        default=None,
        help="Ignored; retained for backward compatibility with run_param_matrix.py.",
    )
    parser.add_argument(
        "--family",
        default="BASE",
        help="Parameter family label (BASE/A/B/C/D). Determines the family_<FAMILY>/ subdirectory.",
    )
    parser.add_argument(
        "--family-run-dir",
        default=None,
        help=(
            "Pre-created family/run<N>/ directory (passed by run_param_matrix.py). "
            "When set, params_<slug>/ is placed directly inside this directory and "
            "the --run-root auto-creation logic is skipped."
        ),
    )
    parser.add_argument(
        "--params-slug",
        default=None,
        help="Explicit params directory name, e.g. params_5_2000000.",
    )
    parser.add_argument("--max-bundle-artifacts", type=int, default=None)
    parser.add_argument("--max-bundle-bytes", type=int, default=None)
    parser.add_argument("--expert-temperature", type=float, default=None)
    parser.add_argument("--design-temperature", type=float, default=None)
    parser.add_argument("--coder-temperature", type=float, default=None)
    parser.add_argument("--review-temperature", type=float, default=None)
    parser.add_argument("--design-max-tokens", type=int, default=None)
    parser.add_argument("--coder-max-tokens", type=int, default=None)
    parser.add_argument("--max-review-loops", type=int, default=None)
    parser.add_argument("--max-design-validation-loops", type=int, default=None)
    parser.add_argument(
        "--run-root",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Root directory for all run output. Ignored when --family-run-dir is set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing params run. Re-runs the first incomplete endpoint from scratch.",
    )
    args = parser.parse_args()

    endpoints = list(DEFAULT_ENDPOINTS)

    family = args.family.upper()
    params_slug = args.params_slug or _auto_params_slug(family, args)

    if args.family_run_dir is not None:
        family_run_dir = Path(args.family_run_dir).resolve()
        family_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_root = Path(args.run_root).resolve()
        family_run_dir = _next_family_run_dir(run_root, family)

    params_dir = family_run_dir / params_slug
    if args.resume:
        params_dir.mkdir(parents=True, exist_ok=True)
    else:
        params_dir.mkdir(parents=True, exist_ok=False)
    (params_dir / "runtime").mkdir(parents=True, exist_ok=True)

    build_smoketest_config, _, _ = _load_smoketest_api()
    cfg = build_smoketest_config(args.model_name)
    cfg = _apply_overrides(cfg, args, params_dir)

    summaries: list[dict] = []
    pending_endpoints = list(endpoints)
    if args.resume:
        summaries, pending_endpoints, is_complete = _resume_results_and_pending_endpoints(params_dir, endpoints)
        if is_complete:
            _write_run_outputs(
                params_dir,
                family=family,
                family_run_dir=family_run_dir,
                params_slug=params_slug,
                page_url=args.page_url,
                endpoints=endpoints,
                cfg=cfg,
                summaries=summaries,
            )
            print(f"Run already complete: {params_dir}")
            return 0

    for endpoint in pending_endpoints:
        print(f"=== PARAM TUNING {family_run_dir.name}/{params_slug}: {endpoint} ===")
        result = await _run_endpoint(endpoint, cfg, params_dir, args.page_url)
        if _RATE_LIMIT_MARKER in result.get("review_summary", ""):
            # Write progress WITHOUT the failed result so --resume restarts
            # from this endpoint rather than skipping it.
            _write_progress(
                params_dir,
                family=family,
                family_run_dir=family_run_dir,
                params_slug=params_slug,
                page_url=args.page_url,
                endpoints=endpoints,
                cfg=cfg,
                results=summaries,
                complete=False,
            )
            print(
                f"Rate limit detected in result for {endpoint} "
                f"({params_slug}) — stopping run. Resume with --resume when quota recovers.",
                file=sys.stderr,
            )
            return _EXIT_RATE_LIMITED
        summaries.append(result)
        _write_progress(
            params_dir,
            family=family,
            family_run_dir=family_run_dir,
            params_slug=params_slug,
            page_url=args.page_url,
            endpoints=endpoints,
            cfg=cfg,
            results=summaries,
            complete=False,
        )

    _write_run_outputs(
        params_dir,
        family=family,
        family_run_dir=family_run_dir,
        params_slug=params_slug,
        page_url=args.page_url,
        endpoints=endpoints,
        cfg=cfg,
        summaries=summaries,
    )
    _write_progress(
        params_dir,
        family=family,
        family_run_dir=family_run_dir,
        params_slug=params_slug,
        page_url=args.page_url,
        endpoints=endpoints,
        cfg=cfg,
        results=summaries,
        complete=True,
    )

    print(f"Saved run to:  {params_dir}")
    print(f"Config file:   {params_dir / _RUN_CONFIG_FILE}")
    print(f"Preview cmds:  {params_dir / _PREVIEW_COMMANDS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
