# IRONMAN 70.3 Finish Time Prediction — A0 Scientific Poster Instructions

This document explains where the editable A0 poster source files and generated high-resolution figures are stored, how to regenerate the A0 PDF, how to verify its exact physical dimensions (`841 × 1189 mm`), and how to customize text or charts.

---

## 1. Poster File Locations

All poster source scripts, 300 DPI figures, and the exported A0 PDF reside inside `bibi_demo_poster/poster/`:

* **Editable Python Generator Source:** `bibi_demo_poster/poster/generate_poster_a0.py`
* **Final Exported A0 PDF:** `bibi_demo_poster/poster/IRONMAN_70_3_Finish_Time_Poster_A0.pdf`
* **High-Resolution (300 DPI) Generated Figures:**
  * `bibi_demo_poster/poster/figures/fig1_cohort_splits.png` — Annual cohort growth (2004–2023, $N = 1,152,658$) and Evaluation I / II temporal split architecture.
  * `bibi_demo_poster/poster/figures/fig2_mae_progression.png` — Progressive MAE reduction across `C0 → C1 → C2` for `B0`, `B2`, `LR`, `RF`, and `CatBoost_ref` in Evaluation I and Evaluation II.
  * `bibi_demo_poster/poster/figures/fig3_p90_trajectory.png` — 90th-percentile absolute error (`P90`) tail-risk contraction across checkpoints in Evaluation II.

---

## 2. How to Regenerate the A0 Poster PDF

Because `generate_poster_a0.py` reads directly from the repository's verified CSV files (`metrics/all_models.csv`, `metrics/cohort.csv`, `metrics/example.csv`, `config/run.json`), running the script automatically regenerates all three 300 DPI figures and compiles the single-page A0 portrait PDF.

In **Windows PowerShell**, run:

```powershell
cd "c:\Users\Biiiii\ironman-70-3-finish-time"
python bibi_demo_poster/poster/generate_poster_a0.py
```

---

## 3. How to Verify Exact A0 Dimensions (`841 × 1189 mm`)

An ISO A0 portrait sheet measures exactly **`841.0 mm × 1189.0 mm`**, which corresponds to **`2383.94 pt × 3370.39 pt`** at 72 points per inch (`1 mm = 2.83464567 pt`).

Run the following PowerShell one-liner using `pypdf` to verify page count, exact dimensions in millimeters, and text extractability:

```powershell
python -c "import pypdf; r = pypdf.PdfReader('bibi_demo_poster/poster/IRONMAN_70_3_Finish_Time_Poster_A0.pdf'); b = r.pages[0].mediabox; pt_to_mm = 25.4 / 72.0; print(f'Pages: {len(r.pages)} | Width: {float(b.width)*pt_to_mm:.2f} mm | Height: {float(b.height)*pt_to_mm:.2f} mm')"
```

Expected output:

```text
Pages: 1 | Width: 841.00 mm | Height: 1189.00 mm
```

---

## 4. How to Make Final Edits to the Poster

1. **Updating Author Names or Institutional Details:**
   * Open `bibi_demo_poster/poster/generate_poster_a0.py`.
   * Locate the `HEADER BANNER` section inside `build_a0_poster()` and update the `Authors:` or `Affiliation:` string calls to `c.drawString(...)`.
2. **Adjusting Colors or Typography:**
   * The Neubrutalist palette constants (`C_CREAM = #FFFDF5`, `C_C0 = #FFD23F`, `C_C1 = #74B9FF`, `C_C2 = #88D498`, `C_CORAL = #FF6B6B`, `C_BLACK = #000000`) are defined at the top of `generate_poster_a0.py`.
3. **Editing Section Copy or Tables:**
   * Each of the 9 modular cards (`1. Introduction`, `2. Research Objectives`, `3. Unified Dataset & QC`, `4. Multi-Checkpoint Methodology`, `5. Experimental Setup`, `6. Progressive Error Visualization`, `7. Experimental Results`, `8. Offline Demo & Real Test Replay`, `9. Conclusions, Limitations & References`) is clearly labeled inside `build_a0_poster()`.
   * After making changes, re-run `python bibi_demo_poster/poster/generate_poster_a0.py`.
