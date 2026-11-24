"""
Scientific Validation & Integrity Test Suite for IRONMAN 70.3 Offline Demo.

Executes and reports on:
1. Model Tests (all 6 CatBoost models: I_C0, I_C1, I_C2, II_C0, II_C1, II_C2)
2. Baseline Tests (all 6 baseline lookup files: I_C0..II_C2)
3. Prediction Consistency & Reference Verification against metrics/example.csv
4. Real Test Replay Data Integrity (all 10 example records, model coverage, hidden actual logic)
5. Offline Self-Sufficiency & Read-Only Path Verification
6. Git Safety & Non-Modifiability Audit
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR.parent
REPO_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(DEMO_DIR))

from inference import (
    IronmanInferenceEngine,
    format_seconds_hms,
    hms_to_seconds,
    validate_athlete_inputs,
)


def run_all_tests() -> Dict[str, Any]:
    engine = IronmanInferenceEngine(REPO_ROOT)
    results: List[Dict[str, Any]] = []

    def record(category: str, name: str, passed: bool, details: str) -> None:
        results.append(
            {
                "category": category,
                "name": name,
                "passed": bool(passed),
                "details": details,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {category} :: {name} -> {details}")

    # =========================================================================
    # 5.1 MODEL TESTS (All 6 CatBoost models)
    # =========================================================================
    expected_features = {
        "C0": ["gender", "ageband", "countryiso", "eventlocation", "eventyear"],
        "C1": ["gender", "ageband", "countryiso", "eventlocation", "eventyear", "swimtime"],
        "C2": [
            "gender",
            "ageband",
            "countryiso",
            "eventlocation",
            "eventyear",
            "swimtime",
            "biketime",
        ],
    }

    for split in ["I", "II"]:
        for cp in ["C0", "C1", "C2"]:
            cb = engine.get_model(split, cp)
            feats = list(cb.feature_names_)
            cat_idx = list(cb.get_cat_feature_indices())
            trees = int(cb.tree_count_)

            ok_meta = (
                feats == expected_features[cp]
                and cat_idx == [0, 2, 3]
                and trees == 120
            )
            record(
                "5.1 Model Tests",
                f"{split}_{cp}.cbm metadata & feature ordering",
                ok_meta,
                f"features={feats}, cat_indices={cat_idx}, tree_count={trees}",
            )

            # Test prediction reproducibility & units (seconds)
            p1 = engine.predict_checkpoint(
                split=split,
                checkpoint=cp,
                gender="F",
                agegroup="35-39",
                countryiso="US",
                eventlocation="IRONMAN 70.3 Florida",
                eventyear=2022 if split == "II" else 2019,
                swimtime_seconds=2100.0 if cp in {"C1", "C2"} else None,
                biketime_seconds=9000.0 if cp == "C2" else None,
            )
            p2 = engine.predict_checkpoint(
                split=split,
                checkpoint=cp,
                gender="F",
                agegroup="35-39",
                countryiso="US",
                eventlocation="IRONMAN 70.3 Florida",
                eventyear=2022 if split == "II" else 2019,
                swimtime_seconds=2100.0 if cp in {"C1", "C2"} else None,
                biketime_seconds=9000.0 if cp == "C2" else None,
            )
            pred_val = p1["CatBoost_ref"]
            ok_pred = (
                np.isfinite(pred_val)
                and 10000.0 < pred_val < 40000.0
                and pred_val == p2["CatBoost_ref"]
            )
            record(
                "5.1 Model Tests",
                f"{split}_{cp}.cbm deterministic inference & unit range",
                ok_pred,
                f"pred={pred_val:.4f}s ({format_seconds_hms(pred_val)}), reproducible={pred_val == p2['CatBoost_ref']}",
            )

    # Test preprocessing edge cases: Namibia 'NA' and whitespace trimming / empty string '__MISSING__'
    df_na = engine.prepare_feature_dataframe(
        checkpoint="C0",
        gender=" M ",
        agegroup="30-34",
        countryiso=" NA ",
        eventlocation="  ",
        eventyear=2022,
    )
    row_na = df_na.iloc[0].to_dict()
    ok_pre = (
        row_na["gender"] == "M"
        and row_na["ageband"] == 30
        and row_na["countryiso"] == "NA"
        and row_na["eventlocation"] == "__MISSING__"
    )
    record(
        "5.1 Model Tests",
        "Preprocessing: Namibia 'NA' preservation & whitespace/__MISSING__",
        ok_pre,
        f"processed_row={row_na}",
    )

    # Test input validation rules
    v_ok, v_err = validate_athlete_inputs(
        "C0", "M", "35-39", "US", "IRONMAN 70.3 Florida", 2022, engine.agegroups
    )
    v_bad_c1, v_err_c1 = validate_athlete_inputs(
        "C1", "M", "35-39", "US", "IRONMAN 70.3 Florida", 2022, engine.agegroups, swimtime_seconds=0
    )
    v_bad_c2, v_err_c2 = validate_athlete_inputs(
        "C2", "M", "35-39", "US", "IRONMAN 70.3 Florida", 2022, engine.agegroups, swimtime_seconds=2000, biketime_seconds=-10
    )
    record(
        "5.1 Model Tests",
        "Input validation for C0, C1, C2",
        v_ok and (not v_bad_c1) and (not v_bad_c2),
        f"C0_valid={v_ok}, C1_zero_swim_rejected={not v_bad_c1}, C2_neg_bike_rejected={not v_bad_c2}",
    )

    # =========================================================================
    # 5.2 BASELINE TESTS (All 6 baseline CSV files)
    # =========================================================================
    for split in ["I", "II"]:
        for cp in ["C0", "C1", "C2"]:
            btable = engine.get_baseline_table(split, cp)
            levels = set(btable["level"].unique())
            n_rows = len(btable)
            min_size = int(btable["size"].min())
            ok_btable = (
                levels == {"all", "gender", "gender+agegroup"}
                and n_rows == 25
                and min_size >= engine.baseline_min_group
            )
            record(
                "5.2 Baseline Tests",
                f"{split}_{cp}_baseline.csv schema & group size threshold",
                ok_btable,
                f"rows={n_rows}, levels={sorted(levels)}, min_size={min_size} (>= {engine.baseline_min_group})",
            )

            # Test B0, B1, B2 calculation
            b_res = engine.predict_baselines(
                split=split,
                checkpoint=cp,
                gender="M",
                agegroup="18-24",
                swimtime_seconds=1573.0 if cp in {"C1", "C2"} else None,
                biketime_seconds=8497.0 if cp == "C2" else None,
            )
            if cp == "C0":
                ok_bcalc = (
                    b_res["B1"] == b_res["B2"]
                    and b_res["known_split_seconds"] == 0.0
                    and b_res["matched_level"] == "gender+agegroup"
                )
            elif cp == "C1":
                ok_bcalc = (
                    "B1" not in b_res
                    and np.isclose(
                        b_res["B2"],
                        1573.0 + b_res["median_remainder_seconds"],
                    )
                )
            else:
                ok_bcalc = (
                    "B1" not in b_res
                    and np.isclose(
                        b_res["B2"],
                        1573.0 + 8497.0 + b_res["median_remainder_seconds"],
                    )
                )
            record(
                "5.2 Baseline Tests",
                f"{split}_{cp} baseline methodology (B0/B1/B2)",
                ok_bcalc,
                f"B0={b_res['B0']}s, B2={b_res['B2']}s, level={b_res['matched_level']}",
            )

    # =========================================================================
    # 5.3 PREDICTION CONSISTENCY AGAINST metrics/example.csv & metrics/all_models.csv
    # =========================================================================
    ex_df = engine.load_example_predictions()
    all_metrics_df = engine.load_all_models_metrics()

    # Check B0 consistency in example.csv (Evaluation II)
    ii_b0_expected = float(
        engine.get_baseline_table("II", "C0")
        .query("level == 'all'")["median"]
        .iloc[0]
    )
    ex_b0_vals = ex_df.query("model == 'B0'")["pred_finish_seconds"].unique()
    ok_b0_match = len(ex_b0_vals) == 1 and np.isclose(ex_b0_vals[0], ii_b0_expected)
    record(
        "5.3 Prediction Consistency",
        "Evaluation II B0 in example.csv matches II_C0_baseline.csv",
        ok_b0_match,
        f"example.csv B0={ex_b0_vals[0]}s vs II_C0_baseline.csv all median={ii_b0_expected}s",
    )

    # Check B1==B2 at C0 across all 10 athletes in example.csv
    c0_b1 = (
        ex_df.query("checkpoint == 'C0' and model == 'B1'")
        .set_index("resultid")["pred_finish_seconds"]
    )
    c0_b2 = (
        ex_df.query("checkpoint == 'C0' and model == 'B2'")
        .set_index("resultid")["pred_finish_seconds"]
    )
    ok_b1_b2 = c0_b1.equals(c0_b2)
    record(
        "5.3 Prediction Consistency",
        "At C0, B1 equals B2 for all 10 test athletes in example.csv",
        ok_b1_b2,
        f"max_abs_diff={(c0_b1 - c0_b2).abs().max():.8f}s",
    )

    # Check that every C0 B1/B2 value in example.csv is an exact median in II_C0_baseline.csv
    ii_c0_medians = set(engine.get_baseline_table("II", "C0")["median"].to_list())
    ok_c0_in_table = all(val in ii_c0_medians for val in c0_b1.to_list())
    record(
        "5.3 Prediction Consistency",
        "All C0 B1/B2 predictions in example.csv exist in II_C0_baseline.csv",
        ok_c0_in_table,
        f"unique_c0_b1_values={sorted(c0_b1.unique())}",
    )

    # Check metrics/all_models.csv progressive MAE reduction for CatBoost_ref
    for split in ["I", "II"]:
        sub = all_metrics_df.query(
            "split == @split and model == 'CatBoost_ref'"
        ).set_index("checkpoint")
        mae_c0 = float(sub.loc["C0", "mae_minutes"])
        mae_c1 = float(sub.loc["C1", "mae_minutes"])
        mae_c2 = float(sub.loc["C2", "mae_minutes"])
        ok_prog = mae_c0 > mae_c1 > mae_c2
        record(
            "5.3 Prediction Consistency",
            f"Evaluation {split} CatBoost_ref progressive MAE reduction (C0 > C1 > C2)",
            ok_prog,
            f"C0={mae_c0:.2f}m -> C1={mae_c1:.2f}m -> C2={mae_c2:.2f}m",
        )

    # =========================================================================
    # 5.4 REPLAY TESTS (metrics/example.csv)
    # =========================================================================
    unique_ids = sorted(ex_df["resultid"].unique().tolist())
    ok_10_ids = len(unique_ids) == 10 and unique_ids == list(range(840060, 840070))
    record(
        "5.4 Replay Tests",
        "All 10 test example records load from metrics/example.csv",
        ok_10_ids,
        f"count={len(unique_ids)}, resultids={unique_ids[0]}..{unique_ids[-1]}",
    )

    models_c0 = sorted(ex_df.query("checkpoint == 'C0'")["model"].unique().tolist())
    models_c1 = sorted(ex_df.query("checkpoint == 'C1'")["model"].unique().tolist())
    models_c2 = sorted(ex_df.query("checkpoint == 'C2'")["model"].unique().tolist())
    ok_models = (
        models_c0 == ["B0", "B1", "B2", "CatBoost_ref", "LR", "RF"]
        and models_c1 == ["B0", "B2", "CatBoost_ref", "LR", "RF"]
        and models_c2 == ["B0", "B2", "CatBoost_ref", "LR", "RF"]
        and len(ex_df) == 160
    )
    record(
        "5.4 Replay Tests",
        "Exact model coverage across checkpoints in example.csv (160 rows)",
        ok_models,
        f"C0={models_c0}; C1={models_c1}; C2={models_c2}; total_rows={len(ex_df)}",
    )

    # Check actual_finish_seconds consistency across all rows for each resultid
    actual_per_id = ex_df.groupby("resultid")["actual_finish_seconds"].nunique()
    ok_actual_unique = (actual_per_id == 1).all()
    record(
        "5.4 Replay Tests",
        "Consistent actual_finish_seconds per resultid across all models & checkpoints",
        bool(ok_actual_unique),
        f"max_unique_actual_per_resultid={int(actual_per_id.max())}",
    )

    # =========================================================================
    # 5.5 OFFLINE SELF-SUFFICIENCY TESTS
    # =========================================================================
    required_local_files = [
        REPO_ROOT / "config" / "run.json",
        REPO_ROOT / "metrics" / "all_models.csv",
        REPO_ROOT / "metrics" / "example.csv",
        REPO_ROOT / "metrics" / "cohort.csv",
        REPO_ROOT / "metrics" / "before_consistency_check.csv",
        REPO_ROOT / "metrics" / "split_consistency.csv",
        REPO_ROOT / "metrics" / "snapshot_comparison.json",
    ]
    for s in ["I", "II"]:
        for c in ["C0", "C1", "C2"]:
            required_local_files.append(REPO_ROOT / "models" / f"{s}_{c}.cbm")
            required_local_files.append(REPO_ROOT / "models" / f"{s}_{c}_baseline.csv")

    all_exist = all(p.exists() for p in required_local_files)
    record(
        "5.5 Offline Tests",
        "All 19 required local research artifacts exist on disk",
        all_exist,
        f"checked={len(required_local_files)} files, all_present={all_exist}",
    )

    # Scan bibi_demo_poster Python files to ensure zero network/cloud/Supabase calls
    forbidden_imports = ["supabase", "requests", "urllib.request", "httpx", "openai"]
    found_forbidden: List[str] = []
    for py_file in DEMO_DIR.rglob("*.py"):
        if py_file.resolve() == Path(__file__).resolve():
            continue
        content = py_file.read_text(encoding="utf-8")
        for imp in forbidden_imports:
            if f"import {imp}" in content or f"from {imp}" in content:
                found_forbidden.append(f"{py_file.name}:{imp}")

    record(
        "5.5 Offline Tests",
        "Zero cloud/API/Supabase network calls in demo source code",
        len(found_forbidden) == 0,
        f"forbidden_imports_found={found_forbidden}",
    )

    # =========================================================================
    # 5.6 GIT INTEGRITY TESTS
    # =========================================================================
    branch_proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    current_branch = branch_proc.stdout.strip()
    record(
        "5.6 Git Integrity",
        "Active Git branch is bibi/demo-poster",
        current_branch == "bibi/demo-poster",
        f"current_branch='{current_branch}'",
    )

    status_proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    status_lines = [
        line.strip() for line in status_proc.stdout.splitlines() if line.strip()
    ]
    outside_changes = [
        line for line in status_lines if "bibi_demo_poster" not in line
    ]
    record(
        "5.6 Git Integrity",
        "Zero modifications outside bibi_demo_poster/",
        len(outside_changes) == 0,
        f"total_status_entries={len(status_lines)}, outside_bibi_demo_poster={outside_changes}",
    )

    diff_proc = subprocess.run(
        ["git", "diff", "--name-only", "main"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    diff_files = [
        line.strip() for line in diff_proc.stdout.splitlines() if line.strip()
    ]
    outside_diff = [f for f in diff_files if not f.startswith("bibi_demo_poster/")]
    record(
        "5.6 Git Integrity",
        "Zero tracked file diffs against main outside bibi_demo_poster/",
        len(outside_diff) == 0,
        f"outside_diff={outside_diff}",
    )

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print(f"\nSUMMARY: {passed_count}/{total_count} scientific & integrity checks PASSED.")

    return {
        "passed_count": passed_count,
        "total_count": total_count,
        "all_passed": passed_count == total_count,
        "results": results,
    }


if __name__ == "__main__":
    summary = run_all_tests()
    if not summary["all_passed"]:
        sys.exit(1)
