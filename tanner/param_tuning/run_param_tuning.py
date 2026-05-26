#!/usr/bin/env python3
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoketest_v2 import build_smoketest_config, generate_bundle_for_path, save_bundle_to_snare_root

DEFAULT_ENDPOINTS = [
    "/boaform/admin/formLogin",
    "/wp-login.php",
    "/api/v1/pods",
    "/containers/json",
    "/.env",
    "/wsman",
]

# Ordered axis keys per family — determines which param values appear in the slug and in what order.
_FAMILY_SLUG_AXES: dict[str, list[str]] = {
    "A": ["max_bundle_artifacts", "max_bundle_bytes"],
    "B": ["design_max_tokens", "coder_max_tokens"],
    "C": ["max_review_loops", "max_design_validation_loops"],
    "D": ["expert_temperature", "design_temperature", "coder_temperature", "review_temperature"],
}


def _slugify_endpoint(endpoint: str) -> str:
    slug = endpoint.strip().lstrip("/") or "root"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug.strip("-") or "root"


def _format_param_value(val) -> str:
    """Format a param value for use in a directory slug (floats use 'p' as decimal separator)."""
    if isinstance(val, float):
        return str(val).replace(".", "p")
    return str(val)


def _auto_params_slug(family: str, args) -> str:
    """
    Build a params slug from the CLI args for the given family.

    Only the axes defined for that family are included, in definition order.
    Falls back to 'params_default' when the family is unrecognised or no
    axis values are present on args.
    """
    axes = _FAMILY_SLUG_AXES.get(family.upper(), [])
    parts = []
    for axis in axes:
        val = getattr(args, axis, None)
        if val is not None:
            parts.append(_format_param_value(val))
    return "params_" + "_".join(parts) if parts else "params_default"


def _next_family_run_dir(run_root: Path, family: str) -> Path:
    """
    Create and return the next runs/family_<FAMILY>/run<N>/ directory.

    The counter increments over existing run<N> siblings so re-running never
    overwrites previous results.
    """
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


async def _run_endpoint(endpoint: str, cfg, params_dir: Path, page_url: str) -> dict:
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
            "and save the results under runs/family_<FAMILY>/run<N>/params_<slug>/."
        )
    )
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--page-url", default="example.com")
    parser.add_argument(
        "--device-endpoint",
        default="/wsman",
        help="Device/protocol endpoint to use in place of the last default endpoint.",
    )
    parser.add_argument(
        "--family",
        default="A",
        help=(
            "Parameter family label (A/B/C/D). Determines the family_<FAMILY>/ "
            "subdirectory and which axes appear in the params slug. "
            "Use BASE for the unmodified baseline configuration."
        ),
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
        help=(
            "Explicit params directory name, e.g. params_5_2000000 "
            "(passed by run_param_matrix.py). "
            "When omitted, auto-generated from the family axes values in --family."
        ),
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
        help=(
            "Root directory for all run output. Ignored when --family-run-dir is set. "
            "Defaults to param_tuning/runs/."
        ),
    )
    args = parser.parse_args()

    endpoints = list(DEFAULT_ENDPOINTS)
    endpoints[-1] = args.device_endpoint

    family = args.family.upper()
    params_slug = args.params_slug or _auto_params_slug(family, args)

    # Resolve the family run directory.
    if args.family_run_dir is not None:
        # Matrix mode: directory already created by run_param_matrix.py.
        family_run_dir = Path(args.family_run_dir).resolve()
        family_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Standalone mode: auto-create runs/family_<FAMILY>/run<N>/.
        run_root = Path(args.run_root).resolve()
        family_run_dir = _next_family_run_dir(run_root, family)

    params_dir = family_run_dir / params_slug
    params_dir.mkdir(parents=True, exist_ok=False)
    (params_dir / "runtime").mkdir(parents=True, exist_ok=True)

    cfg = build_smoketest_config(args.model_name)
    cfg = _apply_overrides(cfg, args, params_dir)

    summaries = []
    for endpoint in endpoints:
        print(f"=== PARAM TUNING {family_run_dir.name}/{params_slug}: {endpoint} ===")
        summaries.append(await _run_endpoint(endpoint, cfg, params_dir, args.page_url))

    run_config = {
        "family": family,
        "family_run_dir": str(family_run_dir),
        "params_slug": params_slug,
        "page_url": args.page_url,
        "endpoints": endpoints,
        "runtime_config": cfg.model_dump(mode="json"),
        "results": summaries,
    }
    (params_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    preview_lines = []
    for summary in summaries:
        preview_lines.append(
            "python preview.py --snare-root {root} --endpoint {endpoint}".format(
                root=params_dir / summary["folder"],
                endpoint=summary["primary_path"],
            )
        )
    (params_dir / "preview_commands.txt").write_text("\n".join(preview_lines) + "\n")

    print(f"Saved run to:  {params_dir}")
    print(f"Config file:   {params_dir / 'run_config.json'}")
    print(f"Preview cmds:  {params_dir / 'preview_commands.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
