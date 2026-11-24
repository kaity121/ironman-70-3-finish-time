# IRONMAN 70.3 Finish Time Prediction — Scientific Validation & Test Report

**Test Suite Script:** `bibi_demo_poster/tests/test_scientific_validation.py`  
**Execution Result:** **39 / 39 Automated Checks PASSED (`100%`)**  
**Browser UI Inspection:** **Verified Live at `http://localhost:8509` (`1440×900`)**

---

## 1. Model Tests (`5.1 Model Tests` — 14/14 Passed)

All six pre-trained CatBoost models (`models/*.cbm`) were loaded and verified against `config/run.json` and `scripts/run.py`:

| Model Artifact | Evaluation Split | Checkpoint | Exact Feature List & Ordering | Cat Indices | Trees | Test Input Sample Output | Status |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `models/I_C0.cbm` | Evaluation I (2004–2016) | `C0` (Before Race) | `['gender', 'ageband', 'countryiso', 'eventlocation', 'eventyear']` | `[0, 2, 3]` | `120` | `22,866.2796 s` (`06:21:06`) | **PASS** |
| `models/I_C1.cbm` | Evaluation I (2004–2016) | `C1` (After Swim) | `['gender', 'ageband', 'countryiso', 'eventlocation', 'eventyear', 'swimtime']` | `[0, 2, 3]` | `120` | `20,963.9865 s` (`05:49:24`) | **PASS** |
| `models/I_C2.cbm` | Evaluation I (2004–2016) | `C2` (After Bike) | `['gender', 'ageband', 'countryiso', 'eventlocation', 'eventyear', 'swimtime', 'biketime']` | `[0, 2, 3]` | `120` | `17,891.9954 s` (`04:58:12`) | **PASS** |
| `models/II_C0.cbm` | Evaluation II (2004–2018) | `C0` (Before Race) | `['gender', 'ageband', 'countryiso', 'eventlocation', 'eventyear']` | `[0, 2, 3]` | `120` | `22,809.2766 s` (`06:20:09`) | **PASS** |
| `models/II_C1.cbm` | Evaluation II (2004–2018) | `C1` (After Swim) | `['gender', 'ageband', 'countryiso', 'eventlocation', 'eventyear', 'swimtime']` | `[0, 2, 3]` | `120` | `20,893.5477 s` (`05:48:14`) | **PASS** |
| `models/II_C2.cbm` | Evaluation II (2004–2018) | `C2` (After Bike) | `['gender', 'ageband', 'countryiso', 'eventlocation', 'eventyear', 'swimtime', 'biketime']` | `[0, 2, 3]` | `120` | `17,714.5321 s` (`04:55:15`) | **PASS** |

* **Preprocessing Verification:** Verified that leading/trailing whitespace is stripped, empty strings are mapped to `'__MISSING__'`, Namibia's ISO code `'NA'` is preserved as string `'NA'` (never converted to `NaN`), and `ageband` is extracted as the integer prefix of `agegroup`.
* **Input Validation Verification:** Verified that `C0` does not accept `swimtime` or `biketime`, `C1` rejects zero/negative `swimtime`, and `C2` rejects zero/negative `swimtime` or `biketime`.

---

## 2. Baseline Tests (`5.2 Baseline Tests` — 12/12 Passed)

All six baseline lookup tables (`models/*_baseline.csv`) were tested against the methodology in `scripts/run.py` (lines 59–75):

| Baseline Artifact | Rows | Hierarchy Levels Present | Minimum Group Size (`>= 30`) | `all` Level Median (`seconds`) | Status |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `models/I_C0_baseline.csv` | `25` | `all`, `gender`, `gender+agegroup` | `79` (F 70–74) | `20,962.0 s` (`05:49:22`) | **PASS** |
| `models/I_C1_baseline.csv` | `25` | `all`, `gender`, `gender+agegroup` | `79` (F 70–74) | `18,622.0 s` (`05:10:22`) | **PASS** |
| `models/I_C2_baseline.csv` | `25` | `all`, `gender`, `gender+agegroup` | `79` (F 70–74) | `7,936.0 s` (`02:12:16`) | **PASS** |
| `models/II_C0_baseline.csv` | `25` | `all`, `gender`, `gender+agegroup` | `137` (F 70–74) | `20,943.0 s` (`05:49:03`) | **PASS** |
| `models/II_C1_baseline.csv` | `25` | `all`, `gender`, `gender+agegroup` | `137` (F 70–74) | `18,602.0 s` (`05:10:02`) | **PASS** |
| `models/II_C2_baseline.csv` | `25` | `all`, `gender`, `gender+agegroup` | `137` (F 70–74) | `7,933.0 s` (`02:12:13`) | **PASS** |

* Verified that `B0` equals the `all` level median of `{split}_C0_baseline.csv`.
* Verified that at `C0`, `B1 == B2`.
* Verified that at `C1` and `C2`, `B2 = known_split_seconds + median_remainder_seconds` using hierarchical lookup (`gender+agegroup` $\rightarrow$ `gender` $\rightarrow$ `all` when `size >= 30`).

---

## 3. Prediction Consistency Tests (`5.3 Prediction Consistency` — 5/5 Passed)

1. **`B0` Exact Match:** All `B0` rows in `metrics/example.csv` (Evaluation II) equal `20,943.0` seconds, matching the `all` row in `models/II_C0_baseline.csv` to `1e-8`.
2. **`B1 == B2` at `C0`:** Across all 10 test athletes in `metrics/example.csv`, `B1` and `B2` at `C0` match with `0.00000000 s` maximum difference.
3. **`C0` Baseline Lookup Verification:** Every `B1`/`B2` value at `C0` in `metrics/example.csv` (`19963.0`, `20176.0`, `20179.0`, `20307.0`, `20502.0`, `22099.0`) is an exact `gender+agegroup` median from `models/II_C0_baseline.csv`.
4. **Progressive MAE Monotonicity:** Verified in `metrics/all_models.csv` that `CatBoost_ref` MAE decreases monotonically across checkpoints in both Evaluation I (`36.37 min` $\rightarrow$ `28.43 min` $\rightarrow$ `14.79 min`) and Evaluation II (`41.72 min` $\rightarrow$ `33.45 min` $\rightarrow$ `15.79 min`).

---

## 4. Real Test Replay Tests (`5.4 Replay Tests` — 3/3 Passed)

1. **10 Authentic Test Records:** Verified that all 10 unique `resultid` records (`840060` through `840069`, `raceid = 774`, `eventyear = 2021`, `split = II`) load cleanly from `metrics/example.csv` (160 total rows).
2. **Saved Method Coverage:** Verified exact saved model sets per checkpoint:
   * `C0`: `['B0', 'B1', 'B2', 'CatBoost_ref', 'LR', 'RF']` (6 models $\times$ 10 athletes = 60 rows)
   * `C1`: `['B0', 'B2', 'CatBoost_ref', 'LR', 'RF']` (5 models $\times$ 10 athletes = 50 rows)
   * `C2`: `['B0', 'B2', 'CatBoost_ref', 'LR', 'RF']` (5 models $\times$ 10 athletes = 50 rows)
3. **Hidden Actual FinishTime Protocol:** Verified in live browser testing that `Actual FinishTime` is locked/hidden by default (`🔒 ACTUAL FINISH TIME: LOCKED / HIDDEN`) and is only revealed after clicking **`REVEAL ACTUAL FINISH TIME`**.

---

## 5. Offline Self-Sufficiency & Git Integrity Tests (`5.5` & `5.6` — 5/5 Passed)

* **Offline Verification:** All 19 local research files (`config/run.json`, 6 `.cbm` models, 6 `*_baseline.csv` files, and 6 `metrics/*` files) load locally from disk. Source code audit confirmed `0` network, cloud, API, or Supabase imports inside `bibi_demo_poster/`.
* **Git Safety Verification:**
  * Active branch confirmed as `bibi/demo-poster`.
  * `git status --porcelain` and `git diff main` confirmed **0 modifications** to any original project file (`models/`, `metrics/`, `config/`, `scripts/`, `v1/`, `README.md`, `requirements.txt`, `.gitignore`).
  * 100% of new files reside inside `bibi_demo_poster/`.

---

## 6. Documented Limitations

1. **No Persisted `.pkl`/`.joblib` Weights for `LR` and `RF`:** In the original repository (`scripts/run.py`), `LinearRegression` and `RandomForestRegressor` are fit in memory to generate `predictions/*.csv.gz` and `metrics/all_models.csv`, while only the 6 CatBoost models (`models/*.cbm`) and 6 baseline lookup CSVs (`models/*_baseline.csv`) are saved to `models/`. Therefore, Mode A (Live Simulation) performs live inference using the 6 saved `CatBoost_ref` models and `B0/B1/B2` baselines without fabricating live `LR` or `RF` predictions, while Mode B (Real Test Replay) displays the authentic saved `LR` and `RF` predictions from `metrics/example.csv`.
2. **Raw ` metrics/example.csv` Schema:** `metrics/example.csv` stores prediction outputs (`resultid, raceid, eventyear, split, checkpoint, model, seed, pred_finish_seconds, actual_finish_seconds`) rather than raw athlete feature columns. In accordance with scientific integrity rules, Mode B displays the exact saved values from `metrics/example.csv` without reconstructing or fabricating missing athlete input columns.
