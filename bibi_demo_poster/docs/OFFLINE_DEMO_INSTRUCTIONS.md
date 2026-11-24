# IRONMAN 70.3 Finish Time Prediction — Offline Demo Instructions

This guide provides step-by-step instructions for installing dependencies, launching, operating, and troubleshooting the **100% offline IRONMAN 70.3 Finish Time Prediction Exhibition Demo** inside `bibi_demo_poster/`.

---

## 1. Required Environment

* **Operating System:** Windows 10 / 11 (or macOS / Linux)
* **Python:** Python 3.12+ (tested on Windows with Python 3.14 & CatBoost 1.2.10)
* **Network Requirement:** **None** after dependencies are installed. The application runs 100% locally and never connects to Supabase, external APIs, or cloud inference servers.
* **Read-Only Original Assets Used:**
  * `config/run.json`
  * `models/I_C0.cbm`, `models/I_C1.cbm`, `models/I_C2.cbm`
  * `models/II_C0.cbm`, `models/II_C1.cbm`, `models/II_C2.cbm`
  * `models/I_C0_baseline.csv` .. `models/II_C2_baseline.csv`
  * `metrics/example.csv`, `metrics/all_models.csv`, `metrics/cohort.csv`, `metrics/before_consistency_check.csv`, `metrics/split_consistency.csv`, `metrics/snapshot_comparison.json`

---

## 2. Dependency Installation (Windows PowerShell)

To preserve the original `requirements.txt` untouched, a dedicated dependency file is provided at `bibi_demo_poster/requirements_demo.txt`.

Open **Windows PowerShell** at the repository root (`c:\Users\Biiiii\ironman-70-3-finish-time`) and run:

```powershell
cd "c:\Users\Biiiii\ironman-70-3-finish-time"
python -m pip install -r bibi_demo_poster/requirements_demo.txt
```

*(Optional)* Verify that all 39 scientific validation and Git integrity tests pass before launching the UI:

```powershell
python bibi_demo_poster/tests/test_scientific_validation.py
```

---

## 3. Launching the Offline Application

From the repository root in **Windows PowerShell**, execute:

```powershell
cd "c:\Users\Biiiii\ironman-70-3-finish-time"
python -m streamlit run bibi_demo_poster/app.py
```

The application will open in your local web browser at `http://localhost:8501` (or the port displayed in your terminal).

---

## 4. How to Use Mode A — Live Race Simulation (`C0 → C1 → C2`)

Mode A allows exhibition visitors to simulate an athlete progressing through an IRONMAN 70.3 race across three checkpoints:

1. **Select Evaluation Cohort:**
   * Use the top dropdown **`EVALUATION COHORT & SAVED MODEL GROUP`** to choose between:
     * **Evaluation II (`models/II_*.cbm`)**: Fit on 2004–2018 ($n = 697,267$), tested on 2021–2023 ($n = 329,366$).
     * **Evaluation I (`models/I_*.cbm`)**: Fit on 2004–2016 ($n = 495,489$), tested on 2019–2020 ($n = 125,912$).
2. **Checkpoint `C0 — BEFORE RACE` (Yellow Theme `#FFD23F`):**
   * Click one of the quick presets (`P1: F 35-39 US`, `P2: M 30-34 VN`, `P3: M 50-54 AT`) or manually enter the 5 pre-race features required by `{split}_C0.cbm`:
     * `gender` (`F` or `M`)
     * `agegroup` (`18-24` to `70-74`, automatically mapped to integer lower-bound `ageband`)
     * `countryiso` (Athlete Country ISO2 code; Namibia `'NA'` is preserved)
     * `eventlocation` (Official IRONMAN 70.3 venue or custom location)
     * `eventyear` (Year of the race)
   * Observe the main predicted **FinishTime** (`HH:MM:SS` and exact seconds) alongside live comparisons against `B0` (global fit median) and `B1`/`B2` (`gender × agegroup` pre-race median).
3. **Checkpoint `C1 — AFTER SWIM` (Sky Blue Theme `#74B9FF`):**
   * Click **`SELECT C1 — AFTER SWIM`** (or **`→ COMPLETE SWIM & ADVANCE TO C1`**).
   * Enter the athlete's actual 1.9 km `SwimTime` in hours, minutes, and seconds (`HH:MM:SS`).
   * The engine loads `{split}_C1.cbm` and `{split}_C1_baseline.csv`, updates the total FinishTime prediction, displays the change relative to `C0` ($\Delta$ vs. `C0`), and shows the estimated time remaining for the Bike + Run stages.
4. **Checkpoint `C2 — AFTER BIKE` (Mint Green Theme `#88D498`):**
   * Click **`SELECT C2 — AFTER BIKE`** (or **`→ COMPLETE BIKE & ADVANCE TO C2`**).
   * Enter the athlete's actual 90 km `BikeTime` in `HH:MM:SS`.
   * The engine loads `{split}_C2.cbm` and `{split}_C2_baseline.csv`, computes the updated FinishTime prediction, displays $\Delta$ vs. `C0` and $\Delta$ vs. `C1`, and calculates the estimated remaining duration for the final **21.1 km Half-Marathon Run (+ Transitions)**.

---

## 5. How to Use Mode B — Real Test Replay (`metrics/example.csv`)

Mode B replays the 10 authentic test records saved in `metrics/example.csv` (`resultid` `840060` through `840069`, Evaluation II, RaceID `774`, Year `2021`):

1. Click **`MODE B: REAL TEST REPLAY (10)`** in the top navigation bar.
2. Select any of the 10 test athletes from **`SELECT SAVED TEST RECORD (resultid)`**.
3. Step through **`REPLAY C0 — BEFORE RACE`**, **`REPLAY C1 — AFTER SWIM`**, and **`REPLAY C2 — AFTER BIKE`** to inspect the original saved predictions across all available methods (`B0`, `B1` at C0, `B2`, `LR`, `RF`, and `CatBoost_ref`).
4. **Revealing the Actual Finish Time:**
   * By default, the athlete's actual finish time is locked (`🔒 ACTUAL FINISH TIME: LOCKED / HIDDEN`).
   * At the final step of your demonstration, click the red button **`REVEAL ACTUAL FINISH TIME`** to unlock the true finish time (`HH:MM:SS` and exact seconds) and view the signed prediction error and absolute error (MAE in minutes) for every model.

---

## 6. How to Reset the Demo

At any point during an exhibition session, click the **`↺ RESET DEMO`** button in the top-right corner of the navigation bar. This immediately restores the default mode (`Mode A: Live Simulation`), resets the checkpoint to `C0`, clears any custom inputs, and re-locks the actual finish time in Mode B.

---

## 7. Common Errors and Solutions

| Symptom / Error | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'catboost'` | CatBoost is not installed in the active Python environment. | Run `python -m pip install -r bibi_demo_poster/requirements_demo.txt`. |
| `INPUT VALIDATION ERROR: SwimTime is outside typical IRONMAN 70.3 bounds` | Entered `SwimTime` is `00:00:00` or outside `00:10:00` – `03:00:00`. | Enter a valid positive 1.9 km swim duration between 10 minutes and 3 hours. |
| `INPUT VALIDATION ERROR: BikeTime is outside typical IRONMAN 70.3 bounds` | Entered `BikeTime` is `00:00:00` or outside `00:40:00` – `07:00:00`. | Enter a valid positive 90 km bike duration between 40 minutes and 7 hours. |
| Port `8501` is already in use | Another Streamlit instance is running. | Specify a custom port: `python -m streamlit run bibi_demo_poster/app.py --server.port 8510`. |
