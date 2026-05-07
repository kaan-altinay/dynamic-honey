#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

from tanner.generator.agentic.models import GeneratorRoleConfig, GeneratorRuntimeConfig
from tanner.generator.agentic.workflow import AgenticBundleGenerator

ROLE_NAMES = ("expert", "design", "coder", "review")


@dataclass
class VerificationResult:
    missing_meta: list[str]
    missing_hash_files: list[str]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_index_page(index_page: str) -> str:
    normalized = index_page.strip() if isinstance(index_page, str) else "/index.html"
    if not normalized:
        normalized = "/index.html"
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized


def normalize_endpoint(raw_endpoint: str, index_page: str) -> str | None:
    endpoint = raw_endpoint.strip()
    if not endpoint or endpoint.startswith("#"):
        return None

    parsed = urlsplit(endpoint)
    if parsed.scheme and parsed.netloc:
        return None

    endpoint = endpoint.split("?", 1)[0]
    endpoint = unquote(endpoint)
    if not endpoint:
        return None
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    if endpoint == "/":
        return index_page
    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]
    return endpoint


def load_endpoints(path: Path, index_page: str) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for line in path.read_text().splitlines():
        normalized = normalize_endpoint(line, index_page)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def load_runtime_config(config_path: Path) -> GeneratorRuntimeConfig:
    config_data = yaml.safe_load(config_path.read_text()) or {}
    generator = _as_dict(config_data.get("GENERATOR"))
    role_defaults = _as_dict(generator.get("role_defaults"))
    configured_roles = _as_dict(generator.get("roles"))

    roles: dict[str, GeneratorRoleConfig] = {}
    for role_name in ROLE_NAMES:
        merged = dict(role_defaults)
        merged.update(_as_dict(configured_roles.get(role_name)))
        roles[role_name] = GeneratorRoleConfig.model_validate(merged)

    return GeneratorRuntimeConfig(
        backend=str(generator.get("backend", "agentic")),
        max_review_loops=int(generator.get("max_review_loops", 2)),
        max_bundle_artifacts=int(generator.get("max_bundle_artifacts", 4)),
        max_bundle_bytes=int(generator.get("max_bundle_bytes", 262_144)),
        checkpoint_path=str(generator.get("checkpoint_path", "/tmp/tanner-agentic-checkpoints.sqlite")),
        graph_recursion_limit=int(generator.get("graph_recursion_limit", 200)),
        review_log_path=str(generator.get("review_log_path", "/tmp/tanner-agentic-review-log.json")),
        enable_live_research=bool(generator.get("enable_live_research", True)),
        max_tool_response_chars=int(generator.get("max_tool_response_chars", 4_000)),
        max_command_output_chars=int(generator.get("max_command_output_chars", 4_000)),
        command_timeout=int(generator.get("command_timeout", 5)),
        max_concurrent_model_calls=int(generator.get("max_concurrent_model_calls", 4)),
        inter_call_delay_seconds=float(generator.get("inter_call_delay_seconds", 0.0)),
        max_rate_limit_retries=int(generator.get("max_rate_limit_retries", 2)),
        default_rate_limit_backoff_seconds=float(generator.get("default_rate_limit_backoff_seconds", 12.0)),
        max_length_limit_retries=int(generator.get("max_length_limit_retries", 2)),
        length_retry_token_increase=int(generator.get("length_retry_token_increase", 800)),
        max_length_retry_tokens=int(generator.get("max_length_retry_tokens", 6_000)),
        roles=roles,
    )


def load_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise ValueError(f"meta.json must contain an object at {meta_path}")
    return meta


def ensure_baseline_meta(meta: dict[str, Any], index_page: str) -> None:
    if index_page not in meta:
        raise ValueError(f"Baseline index entry missing from meta.json: {index_page}")
    if "/status_404" not in meta:
        raise ValueError("Baseline /status_404 entry missing from meta.json")
    if "/" not in meta:
        meta["/"] = meta[index_page]


def write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def write_seedfile(seedfile_path: Path, endpoints: list[str]) -> None:
    seedfile_path.write_text("\n".join(endpoints) + "\n")


def verify_cache(meta: dict[str, Any], page_dir: Path, endpoints: list[str]) -> VerificationResult:
    missing_meta = [endpoint for endpoint in endpoints if endpoint not in meta]

    missing_hash_files: list[str] = []
    for endpoint in endpoints:
        entry = meta.get(endpoint)
        if not isinstance(entry, dict):
            continue
        hash_name = entry.get("hash")
        if not isinstance(hash_name, str) or not hash_name:
            missing_hash_files.append(f"{endpoint} -> <missing hash>")
            continue
        if not (page_dir / hash_name).is_file():
            missing_hash_files.append(f"{endpoint} -> {hash_name}")

    return VerificationResult(missing_meta=missing_meta, missing_hash_files=missing_hash_files)


async def prewarm(args: argparse.Namespace) -> int:
    index_page = _normalize_index_page(args.index_page)

    endpoints_path = Path(args.endpoints_file).resolve()
    generator_config_path = Path(args.generator_config).resolve()
    snare_root = Path(args.snare_root).resolve()
    page_dir = snare_root / "pages" / args.page_url
    meta_path = page_dir / "meta.json"
    seedfile_path = snare_root / "seedfile.txt"

    runtime_config = load_runtime_config(generator_config_path)
    if runtime_config.backend.strip().lower() != "agentic":
        raise ValueError(
            f"GENERATOR.backend in {generator_config_path} is '{runtime_config.backend}', expected 'agentic'"
        )

    endpoints = load_endpoints(endpoints_path, index_page)
    if not endpoints:
        raise ValueError(f"No valid internal endpoints found in {endpoints_path}")

    meta = load_meta(meta_path)
    ensure_baseline_meta(meta, index_page)

    if args.write_seedfile:
        write_seedfile(seedfile_path, endpoints)

    summary = {
        "requested": len(endpoints),
        "skipped": 0,
        "generated": 0,
        "failed": 0,
        "paths_written": 0,
        "paths_deduped": 0,
        "path_collisions": 0,
        "path_collision_skips": 0,
        "path_collision_overwrites": 0,
    }

    generator = AgenticBundleGenerator(runtime_config=runtime_config)
    collisions: list[dict[str, str]] = []

    if not args.check_only:
        for endpoint in endpoints:
            if endpoint in meta and not args.force:
                summary["skipped"] += 1
                continue

            try:
                bundle = await generator.generate_bundle(
                    host=args.host,
                    path=endpoint,
                    site_profile={"index_page": index_page},
                )
            except Exception as error:  # noqa: BLE001
                summary["failed"] += 1
                print(f"[FAIL] {endpoint}: generation raised {error}")
                continue

            try:
                if not bundle.artifacts:
                    raise ValueError("bundle returned no artifacts")

                written_this_endpoint = 0
                for artifact in bundle.artifacts:
                    body = bytes(artifact.body_bytes)
                    digest = hashlib.md5(body).hexdigest()
                    artifact_path = normalize_endpoint(artifact.path, index_page) or endpoint
                    headers = artifact.headers if isinstance(artifact.headers, list) else []

                    existing_entry = meta.get(artifact_path)
                    existing_hash = existing_entry.get("hash") if isinstance(existing_entry, dict) else None
                    if isinstance(existing_hash, str) and existing_hash:
                        if existing_hash == digest:
                            summary["paths_deduped"] += 1
                            continue

                        summary["path_collisions"] += 1
                        collision = {
                            "endpoint": endpoint,
                            "path": artifact_path,
                            "existing_hash": existing_hash,
                            "new_hash": digest,
                        }
                        collisions.append(collision)

                        if not args.force_path_overwrite:
                            summary["path_collision_skips"] += 1
                            print(
                                "[COLLISION] {endpoint}: {path} existing={existing_hash} new={new_hash} (skipped)".format(
                                    **collision
                                )
                            )
                            continue

                        summary["path_collision_overwrites"] += 1
                        print(
                            "[COLLISION] {endpoint}: {path} existing={existing_hash} new={new_hash} (overwritten by --force-path-overwrite)".format(
                                **collision
                            )
                        )

                    (page_dir / digest).write_bytes(body)
                    meta[artifact_path] = {"hash": digest, "headers": headers}
                    written_this_endpoint += 1
                write_meta(meta_path, meta)
                summary["generated"] += 1
                summary["paths_written"] += written_this_endpoint
                print(f"[OK] {endpoint}: wrote {written_this_endpoint} artifact paths")
            except Exception as error:  # noqa: BLE001
                summary["failed"] += 1
                print(f"[FAIL] {endpoint}: persistence failed {error}")

    verification = verify_cache(meta, page_dir, endpoints)

    print("\n=== PREWARM SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"missing_meta_entries: {len(verification.missing_meta)}")
    print(f"missing_hash_files: {len(verification.missing_hash_files)}")

    if verification.missing_meta:
        print("\nMissing meta entries:")
        for endpoint in verification.missing_meta[:30]:
            print(f"  - {endpoint}")

    if verification.missing_hash_files:
        print("\nMissing hash files:")
        for issue in verification.missing_hash_files[:30]:
            print(f"  - {issue}")
    if collisions:
        print("\nPath collisions:")
        for collision in collisions[:50]:
            print(
                "  - endpoint={endpoint} path={path} existing={existing_hash} new={new_hash}".format(
                    **collision
                )
            )
        if len(collisions) > 50:
            print(f"  ... {len(collisions) - 50} more")

    has_failures = (
        summary["failed"] > 0
        or summary["path_collision_skips"] > 0
        or bool(verification.missing_meta)
        or bool(verification.missing_hash_files)
    )


    return 1 if has_failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-generate and verify Snare cache entries offline for a future run.",
    )
    parser.add_argument(
        "--endpoints-file",
        required=True,
        help="Path to newline-separated endpoint list.",
    )
    parser.add_argument(
        "--generator-config",
        default="/home/kaan/dynamic-honey/tanner/tanner/data/config.v1-smoketest.yaml",
        help="Path to Tanner YAML config with GENERATOR settings to mirror smoketest behavior.",
    )
    parser.add_argument(
        "--snare-root",
        default="/home/kaan/snare-data/snare",
        help="Snare template root (contains pages/ and seedfile.txt).",
    )
    parser.add_argument(
        "--page-url",
        default="example.com",
        help="Page directory under <snare-root>/pages.",
    )
    parser.add_argument(
        "--host",
        default="example.com",
        help="Host value passed to generator (context only).",
    )
    parser.add_argument(
        "--index-page",
        default="/index.html",
        help="Index page used for normalization and site_profile.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Do not generate new content; only validate endpoint coverage and hash files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate endpoints even if already present in meta.json.",
    )
    parser.add_argument(
        "--force-path-overwrite",
        action="store_true",
        help=("Allow overwriting existing meta.json path entries when generated artifacts conflict with different hashes. "
              "Default behavior is collision-safe skip with non-zero exit."),
    )
    parser.add_argument(
        "--write-seedfile",
        action="store_true",
        help="Write normalized endpoints into <snare-root>/seedfile.txt.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(prewarm(args))


if __name__ == "__main__":
    raise SystemExit(main())
