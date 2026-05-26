#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

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

FAMILY_VALUES = {
    "A": {
        "description": "Bundle size budget",
        "axes": {
            "max_bundle_artifacts": [5, 8],
            "max_bundle_bytes": [1_048_576, 2_000_000, 4_000_000],
        },
    },
    "B": {
        "description": "Design/Coder token budgets",
        "axes": {
            "design_max_tokens": [1800, 2400, 2800],
            "coder_max_tokens": [1200, 1600, 2200],
        },
    },
    "C": {
        "description": "Review loop depth",
        "axes": {
            "max_review_loops": [1, 2],
            "max_design_validation_loops": [2, 3],
        },
    },
    "D": {
        "description": "Per-role temperatures",
        "axes": {
            "expert_temperature": [0.0, 0.1],
            "design_temperature": [0.0, 0.1, 0.2],
            "coder_temperature": [0.0, 0.1, 0.2],
            "review_temperature": [0.0, 0.1],
        },
    },
}

# Ordered axis keys per family — same ordering used in run_param_tuning.py.
_FAMILY_SLUG_AXES: dict[str, list[str]] = {
    family: list(defn["axes"].keys())
    for family, defn in FAMILY_VALUES.items()
}


def _format_param_value(val) -> str:
    if isinstance(val, float):
        return str(val).replace(".", "p")
    return str(val)


def _params_slug_for_family(family: str, params: dict) -> str:
    """
    Build the params_<slug> directory name for a given family and parameter dict.

    BASE (baseline) always maps to 'params_baseline'.
    Other families use the ordered axis values for that family joined by '_'.
    """
    if family == "BASE":
        return "params_baseline"
    axes = _FAMILY_SLUG_AXES.get(family, [])
    parts = [_format_param_value(params[axis]) for axis in axes]
    return "params_" + "_".join(parts)


def _next_family_run_dir(run_root: Path, family: str) -> Path:
    """
    Create and return the next runs/family_<FAMILY>/run<N>/ directory.

    Increments the run counter over existing siblings so multiple matrix
    executions for the same family never collide.
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


def _combo_key(params: dict) -> tuple:
    return tuple((key, params[key]) for key in sorted(BASELINE))


def build_plan(families: list[str]) -> list[dict]:
    plan: list[dict] = []
    seen: set[tuple] = set()

    baseline_key = _combo_key(BASELINE)
    seen.add(baseline_key)
    plan.append(
        {
            "family": "BASE",
            "family_description": "Baseline smoketest configuration",
            "varied_param": None,
            "params_slug": "params_baseline",
            "params": dict(BASELINE),
        }
    )

    for family in families:
        family_def = FAMILY_VALUES[family]
        for axis_name, axis_values in family_def["axes"].items():
            for value in axis_values:
                if value == BASELINE[axis_name]:
                    continue
                params = dict(BASELINE)
                params[axis_name] = value
                key = _combo_key(params)
                if key in seen:
                    continue
                seen.add(key)
                plan.append(
                    {
                        "family": family,
                        "family_description": family_def["description"],
                        "varied_param": axis_name,
                        "params_slug": _params_slug_for_family(family, params),
                        "params": params,
                    }
                )

    return plan


def print_plan(plan: list[dict]) -> None:
    print(f"Total unique runs: {len(plan)}")
    for idx, entry in enumerate(plan, start=1):
        params = entry["params"]
        varied = entry["varied_param"] or "baseline"
        print(
            "[{idx:02d}] family={family:<4} varied={varied:<28} slug={slug:<30} "
            "artifacts={artifacts} bytes={bytes_:>9} "
            "design_tok={design} coder_tok={coder} "
            "review_loops={review} dv_loops={dv} "
            "t_expert={t_expert} t_design={t_design} t_coder={t_coder} t_review={t_review}".format(
                idx=idx,
                family=entry["family"],
                varied=varied,
                slug=entry["params_slug"],
                artifacts=params["max_bundle_artifacts"],
                bytes_=params["max_bundle_bytes"],
                design=params["design_max_tokens"],
                coder=params["coder_max_tokens"],
                review=params["max_review_loops"],
                dv=params["max_design_validation_loops"],
                t_expert=params["expert_temperature"],
                t_design=params["design_temperature"],
                t_coder=params["coder_temperature"],
                t_review=params["review_temperature"],
            )
        )


def run_plan(plan: list[dict], args) -> None:
    script_path = Path(__file__).resolve().parent / "run_param_tuning.py"
    run_root = Path(args.run_root).resolve()

    # Pre-create one family/run<N>/ directory per family so that all parameter
    # configurations in the same matrix execution share a common run counter.
    family_run_dirs: dict[str, Path] = {}
    for entry in plan:
        family = entry["family"]
        if family not in family_run_dirs:
            family_run_dirs[family] = _next_family_run_dir(run_root, family)

    for idx, entry in enumerate(plan, start=1):
        params = entry["params"]
        family = entry["family"]
        family_run_dir = family_run_dirs[family]
        params_slug = entry["params_slug"]

        cmd = [
            sys.executable,
            str(script_path),
            "--model-name", args.model_name,
            "--page-url", args.page_url,
            "--device-endpoint", args.device_endpoint,
            "--family", family,
            "--family-run-dir", str(family_run_dir),
            "--params-slug", params_slug,
            "--max-bundle-artifacts", str(params["max_bundle_artifacts"]),
            "--max-bundle-bytes", str(params["max_bundle_bytes"]),
            "--expert-temperature", str(params["expert_temperature"]),
            "--design-temperature", str(params["design_temperature"]),
            "--coder-temperature", str(params["coder_temperature"]),
            "--review-temperature", str(params["review_temperature"]),
            "--design-max-tokens", str(params["design_max_tokens"]),
            "--coder-max-tokens", str(params["coder_max_tokens"]),
            "--max-review-loops", str(params["max_review_loops"]),
            "--max-design-validation-loops", str(params["max_design_validation_loops"]),
        ]
        print(f"\n=== RUN {idx}/{len(plan)} family={family} dir={family_run_dir.name}/{params_slug} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a one-parameter-at-a-time V2 parameter tuning matrix."
    )
    parser.add_argument(
        "--families",
        default="A,B,C,D",
        help="Comma-separated subset of families to run (default: A,B,C,D).",
    )
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--page-url", default="example.com")
    parser.add_argument("--device-endpoint", default="/wsman")
    parser.add_argument(
        "--run-root",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Root directory for all run output. Defaults to param_tuning/runs/.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the matrix. Default is plan-only (dry run).",
    )
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Write the generated matrix plan to <run-root>/param_matrix_plan.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = [value.strip().upper() for value in args.families.split(",") if value.strip()]
    invalid = [family for family in families if family not in FAMILY_VALUES]
    if invalid:
        raise SystemExit(f"Unknown family ids: {', '.join(invalid)}")

    plan = build_plan(families)
    print_plan(plan)

    if args.write_plan:
        plan_path = Path(args.run_root).resolve() / "param_matrix_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(f"Plan written to: {plan_path}")

    if args.execute:
        run_plan(plan, args)
    else:
        print("\nPlan only. Re-run with --execute to launch all runs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
