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
            "max_bundle_bytes": [1_000_000, 2_000_000, 4_000_000],
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

DEFAULT_ENDPOINTS = [
    "/boaform/admin/formLogin",
    "/api/v1/pods",
    "/containers/json",
    "/.env",
]
# Must match _EXIT_RATE_LIMITED in run_param_tuning.py.
_EXIT_RATE_LIMITED = 2

_FAMILY_SLUG_AXES: dict[str, list[str]] = {
    family: list(definition["axes"].keys())
    for family, definition in FAMILY_VALUES.items()
}


def _format_param_value(value) -> str:
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value)


def _params_slug_for_family(family: str, params: dict) -> str:
    if family == "BASE":
        return "params_baseline"
    axes = _FAMILY_SLUG_AXES.get(family, [])
    parts = [_format_param_value(params[axis]) for axis in axes]
    return "params_" + "_".join(parts)


def _combo_key(params: dict) -> tuple:
    return tuple((key, params[key]) for key in sorted(BASELINE))


def _family_run_dir(run_root: Path, family: str, run_name: str) -> Path:
    return run_root / f"family_{family}" / run_name


def _next_run_name(run_root: Path) -> str:
    run_root.mkdir(parents=True, exist_ok=True)
    max_n = 0
    pattern = re.compile(r"run(\d+)$")
    for family_dir in run_root.iterdir():
        if not family_dir.is_dir() or not family_dir.name.startswith("family_"):
            continue
        for child in family_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.fullmatch(child.name)
            if match:
                max_n = max(max_n, int(match.group(1)))
    return f"run{max_n + 1}"


def _matrix_manifest_path(run_root: Path, run_name: str) -> Path:
    return run_root / f"matrix_{run_name}.json"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _params_run_complete(params_dir: Path, endpoints: list[str]) -> bool:
    progress = _read_json(params_dir / "run_state.json")
    if progress is None:
        progress = _read_json(params_dir / "run_config.json")
        if progress is None:
            return False
        results = list(progress.get("results", []))
        complete = True
    else:
        results = list(progress.get("results", []))
        complete = bool(progress.get("complete"))

    saved_endpoints = progress.get("endpoints") or endpoints
    if len(results) != len(saved_endpoints):
        return False
    if not complete:
        return False
    return all(summary.get("endpoint") == endpoint for summary, endpoint in zip(results, saved_endpoints))


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
        family_definition = FAMILY_VALUES[family]
        for axis_name, axis_values in family_definition["axes"].items():
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
                        "family_description": family_definition["description"],
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


def _write_matrix_manifest(plan: list[dict], args, run_root: Path, run_name: str, endpoints: list[str]) -> None:
    manifest = {
        "run_name": run_name,
        "run_root": str(run_root),
        "page_url": args.page_url,
        "model_name": args.model_name,
        "families": args.families,
        "endpoints": endpoints,
        "plan": [
            {
                **entry,
                "family_run_dir": str(_family_run_dir(run_root, entry["family"], run_name)),
                "params_dir": str(_family_run_dir(run_root, entry["family"], run_name) / entry["params_slug"]),
            }
            for entry in plan
        ],
    }
    _matrix_manifest_path(run_root, run_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run_plan(plan: list[dict], args, run_name: str) -> None:
    script_path = Path(__file__).resolve().parent / "run_param_tuning.py"
    run_root = Path(args.run_root).resolve()
    endpoints = list(DEFAULT_ENDPOINTS)

    family_run_dirs: dict[str, Path] = {}
    for entry in plan:
        family = entry["family"]
        if family in family_run_dirs:
            continue
        family_run_dir = _family_run_dir(run_root, family, run_name)
        family_run_dir.mkdir(parents=True, exist_ok=args.resume)
        family_run_dirs[family] = family_run_dir

    _write_matrix_manifest(plan, args, run_root, run_name, endpoints)

    for idx, entry in enumerate(plan, start=1):
        params = entry["params"]
        family = entry["family"]
        family_run_dir = family_run_dirs[family]
        params_slug = entry["params_slug"]
        params_dir = family_run_dir / params_slug

        if args.resume and _params_run_complete(params_dir, endpoints):
            print(f"\n=== SKIP {idx}/{len(plan)} family={family} dir={family_run_dir.name}/{params_slug} (complete) ===")
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--model-name",
            args.model_name,
            "--page-url",
            args.page_url,
            "--family",
            family,
            "--family-run-dir",
            str(family_run_dir),
            "--params-slug",
            params_slug,
            "--max-bundle-artifacts",
            str(params["max_bundle_artifacts"]),
            "--max-bundle-bytes",
            str(params["max_bundle_bytes"]),
            "--expert-temperature",
            str(params["expert_temperature"]),
            "--design-temperature",
            str(params["design_temperature"]),
            "--coder-temperature",
            str(params["coder_temperature"]),
            "--review-temperature",
            str(params["review_temperature"]),
            "--design-max-tokens",
            str(params["design_max_tokens"]),
            "--coder-max-tokens",
            str(params["coder_max_tokens"]),
            "--max-review-loops",
            str(params["max_review_loops"]),
            "--max-design-validation-loops",
            str(params["max_design_validation_loops"]),
        ]
        if args.resume:
            cmd.append("--resume")

        print(f"\n=== RUN {idx}/{len(plan)} family={family} dir={family_run_dir.name}/{params_slug} ===")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == _EXIT_RATE_LIMITED:
            print(
                f"Matrix stopped: rate limit hit during {family}/{params_slug}. "
                f"Resume with --resume --run-name {run_name} when quota recovers.",
                file=sys.stderr,
            )
            sys.exit(_EXIT_RATE_LIMITED)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"\n=== MATRIX COMPLETE: {run_name} ({len(plan)} run(s)) ===")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a one-parameter-at-a-time V2 parameter tuning matrix.")
    parser.add_argument("--families", default="A,B,C,D", help="Comma-separated subset of families to run (default: A,B,C,D)")
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--page-url", default="example.com")
    parser.add_argument("--device-endpoint", default=None, help="Ignored; retained for backward compatibility.")
    parser.add_argument("--run-root", default=str(Path(__file__).resolve().parent / "runs"))
    parser.add_argument("--run-name", default=None, help="Matrix execution name, e.g. run2. Auto-assigned when omitted.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted matrix execution. Requires --run-name.")
    parser.add_argument("--execute", action="store_true", help="Actually execute the matrix. Default is plan-only.")
    parser.add_argument("--write-plan", action="store_true", help="Write the generated matrix plan to param_matrix_plan.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = [value.strip().upper() for value in args.families.split(",") if value.strip()]
    invalid = [family for family in families if family not in FAMILY_VALUES]
    if invalid:
        raise SystemExit(f"Unknown family ids: {', '.join(invalid)}")
    if args.resume and not args.run_name:
        raise SystemExit("--resume requires --run-name")

    plan = build_plan(families)
    print_plan(plan)

    run_root = Path(args.run_root).resolve()
    if args.write_plan:
        plan_path = run_root / "param_matrix_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(f"Plan written to: {plan_path}")

    if args.execute:
        run_name = args.run_name or _next_run_name(run_root)
        _write_matrix_manifest(plan, args, run_root, run_name, list(DEFAULT_ENDPOINTS))
        print(f"Matrix run name: {run_name}")
        run_plan(plan, args, run_name)
    else:
        print("\nPlan only. Re-run with --execute to launch all runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
