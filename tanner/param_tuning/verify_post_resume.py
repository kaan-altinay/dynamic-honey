import json
from pathlib import Path

DEFAULT_ENDPOINTS = ["/boaform/admin/formLogin", "/api/v1/pods", "/containers/json", "/.env"]
SLUGS = {
    "/boaform/admin/formLogin": "boaform-admin-formLogin",
    "/api/v1/pods": "api-v1-pods",
    "/containers/json": "containers-json",
    "/.env": ".env",
}

# (dir, min_idx) from before the resume, so we know exactly which endpoints
# were regenerated this time and should be re-checked.
TARGETS = [
    ("family_BASE/run2/params_baseline", 1),
    ("family_BASE/run3/params_baseline", 2),
    ("family_BASE/run4/params_baseline", 2),
    ("family_A/run2/params_5_2000000", 2),
    ("family_A/run2/params_8_1048576", 3),
    ("family_A/run2/params_8_4000000", 2),
    ("family_B/run3/params_2800_1200", 2),
    ("family_B/run3/params_2800_2200", 2),
    ("family_C/run4/params_1_3", 1),
    ("family_C/run4/params_2_2", 3),
    ("family_D/run4/params_0p0_0p0_0p0_0p0", 2),
    ("family_D/run4/params_0p0_0p0_0p1_0p1", 3),
    ("family_D/run4/params_0p0_0p0_0p2_0p0", 1),
    ("family_D/run4/params_0p0_0p1_0p1_0p0", 1),
    ("family_D/run4/params_0p0_0p2_0p1_0p0", 2),
    ("family_D/run4/params_0p1_0p0_0p1_0p0", 2),
]
UNTOUCHED_CONTROLS = [
    "family_B/run3/params_1800_1600",
    "family_B/run3/params_2400_1600",
]

root = Path("runs")

def bug_signatures(rs, slug):
    found = []
    if slug == "containers-json" and "input_type=list" in rs:
        found.append("bug1_list_dict")
    if slug == "api-v1-pods" and "extra_forbidden" in rs:
        found.append("bug4_extra_forbidden")
    if slug == ".env" and "config_theft bundles must include a supporting" in rs:
        found.append("bug2_config_secret_routing")
    if slug == ".env" and "got backup_manifest)" in rs:
        found.append("bug3_backup_manifest_kind")
    return found

print("=== Re-resumed dirs: completeness + regenerated-endpoint health ===")
all_clean = True
for rel, min_idx in TARGETS:
    pdir = root / rel
    state = json.loads((pdir / "run_state.json").read_text())
    complete = state.get("complete")
    results_count = len(state.get("results", []))
    regenerated = DEFAULT_ENDPOINTS[min_idx:]

    print(f"\n[{rel}]  complete={complete} results_count={results_count}  regenerated_this_round={regenerated}")
    for ep in regenerated:
        slug = SLUGS[ep]
        summ_path = pdir / slug / "bundle_summary.json"
        if not summ_path.exists():
            print(f"    ! MISSING bundle_summary.json for {ep}")
            all_clean = False
            continue
        summ = json.loads(summ_path.read_text())
        rs = summ.get("review_summary") or ""
        bugs = bug_signatures(rs, slug)
        used_fallback = summ.get("used_fallback")
        artifact_count = summ.get("artifact_count")
        status = "BUG STILL PRESENT" if bugs else "ok"
        if bugs:
            all_clean = False
        print(f"    {ep:<25} artifacts={artifact_count} used_fallback={used_fallback} status={status} {bugs}")
        print(f"        review_summary: {rs[:160]}")

print("\n=== Untouched controls (should be byte-identical, unaffected) ===")
for rel in UNTOUCHED_CONTROLS:
    pdir = root / rel
    state = json.loads((pdir / "run_state.json").read_text())
    print(f"[{rel}] complete={state.get('complete')} results_count={len(state.get('results', []))}")

print("\n=== OVERALL:", "ALL CLEAN" if all_clean else "ISSUES REMAIN", "===")
