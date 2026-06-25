import json
from pathlib import Path

DEFAULT_ENDPOINTS = ["/boaform/admin/formLogin", "/api/v1/pods", "/containers/json", "/.env"]
SLUGS = {
    "/boaform/admin/formLogin": "boaform-admin-formLogin",
    "/api/v1/pods": "api-v1-pods",
    "/containers/json": "containers-json",
    "/.env": ".env",
}

TARGETS = [
    "family_BASE/run2/params_baseline", "family_BASE/run3/params_baseline", "family_BASE/run4/params_baseline",
    "family_A/run2/params_5_2000000", "family_A/run2/params_8_1048576", "family_A/run2/params_8_4000000",
    "family_A/run2/params_8_1000000",
    "family_B/run3/params_2800_1200", "family_B/run3/params_2800_2200",
    "family_C/run4/params_1_3", "family_C/run4/params_2_2",
    "family_D/run4/params_0p0_0p0_0p0_0p0", "family_D/run4/params_0p0_0p0_0p1_0p1",
    "family_D/run4/params_0p0_0p0_0p2_0p0", "family_D/run4/params_0p0_0p1_0p1_0p0",
    "family_D/run4/params_0p0_0p2_0p1_0p0", "family_D/run4/params_0p1_0p0_0p1_0p0",
]

root = Path("runs")
for rel in TARGETS:
    pdir = root / rel
    state_path = pdir / "run_state.json"
    if not state_path.exists():
        print(f"[{rel}] NO run_state.json")
        continue
    state = json.loads(state_path.read_text())
    print(f"\n[{rel}] complete={state.get('complete')} results={len(state.get('results', []))}")
    for ep, slug in SLUGS.items():
        summ_path = pdir / slug / "bundle_summary.json"
        if not summ_path.exists():
            continue
        summ = json.loads(summ_path.read_text())
        rs = summ.get("review_summary") or ""
        used_fallback = summ.get("used_fallback")
        artifact_count = summ.get("artifact_count")
        has_doc_dict_type = "content_model.document" in rs and "dict_type" in rs
        has_extra_forbidden = "extra_forbidden" in rs
        has_top_level_sibling = has_extra_forbidden and "content_model." not in rs
        has_config_theft_gap = "config_theft bundles must include a supporting" in rs
        has_backup_manifest_kind = "got backup_manifest)" in rs
        flags = []
        if has_doc_dict_type:
            flags.append("DOC_LIST_VS_DICT(should be fixed)")
        if has_top_level_sibling:
            flags.append("TOP_LEVEL_SIBLING_extra_forbidden(NOT covered by fix)")
        elif has_extra_forbidden:
            flags.append("NESTED_extra_forbidden(should be fixed)")
        if has_config_theft_gap:
            flags.append("CONFIG_THEFT_GAP(should be fixed)")
        if has_backup_manifest_kind:
            flags.append("BACKUP_MANIFEST_KIND(should be fixed for .json paths)")
        print(f"    {ep:<25} artifacts={artifact_count} used_fallback={used_fallback} flags={flags or ['clean']}")
