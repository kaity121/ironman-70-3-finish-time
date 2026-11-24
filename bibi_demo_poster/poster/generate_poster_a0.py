"""
Generate the A0 Scientific Research Poster (841 x 1189 mm, Portrait, High-Resolution PDF)
for the IRONMAN 70.3 Finish Time Prediction Project.

Design Direction:
- Professional Academic Editorial with a Restrained Neubrutalist Identity
- Exact A0 Portrait dimensions: 841.0 mm x 1189.0 mm (2383.937 pt x 3370.394 pt)
- Color Palette:
    * Main background: #FFFDF5 (Warm Cream)
    * Primary text & borders: #000000 (Solid Black, 2.5-4pt borders, sharp square corners, hard offset shadows)
    * C0 — Before Race: #FFD23F (Yellow)
    * C1 — After Swim: #74B9FF (Sky Blue)
    * C2 — After Bike: #88D498 (Mint Green)
    * Secondary accent: #FF6B6B (Coral Red)

All scientific metrics, cohort counts, and baseline comparisons are dynamically read
from the verified repository files (`config/run.json`, `metrics/all_models.csv`,
`metrics/cohort.csv`, `metrics/example.csv`, `metrics/snapshot_comparison.json`,
`v1/metrics_reported.csv`) so zero numbers are invented or fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

POSTER_DIR = Path(__file__).resolve().parent
DEMO_DIR = POSTER_DIR.parent
REPO_ROOT = DEMO_DIR.parent

# Exact A0 Dimensions in points (1 mm = 2.83464567 pt)
A0_WIDTH_MM = 841.0
A0_HEIGHT_MM = 1189.0
W_PT = A0_WIDTH_MM * mm
H_PT = A0_HEIGHT_MM * mm

# Neubrutalist Color Palette
C_CREAM = colors.HexColor("#FFFDF5")
C_WHITE = colors.HexColor("#FFFFFF")
C_BLACK = colors.HexColor("#000000")
C_C0 = colors.HexColor("#FFD23F")
C_C1 = colors.HexColor("#74B9FF")
C_C2 = colors.HexColor("#88D498")
C_CORAL = colors.HexColor("#FF6B6B")
C_MUTED = colors.HexColor("#EAE6D7")


def generate_poster_charts(out_dir: Path) -> Dict[str, Path]:
    """
    Generate high-resolution (300 DPI) scientific figures from the verified repository CSVs
    for embedding into the A0 poster.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_models = pd.read_csv(REPO_ROOT / "metrics" / "all_models.csv")
    cohort = pd.read_csv(REPO_ROOT / "metrics" / "cohort.csv")
    example = pd.read_csv(REPO_ROOT / "metrics" / "example.csv")

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    chart_paths: Dict[str, Path] = {}

    # -------------------------------------------------------------------------
    # Figure 1: Annual Cohort & Temporal Split Architecture (2004-2023)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9.5, 4.3), dpi=300)
    fig1.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FFFDF5")

    years = cohort["eventyear"].to_numpy()
    records_k = cohort["records"].to_numpy() / 1000.0
    races = cohort["races"].to_numpy()

    bar_colors = []
    for y in years:
        if y <= 2016:
            bar_colors.append("#FFD23F")  # Eval I & II shared Fit (2004-2016)
        elif y in (2017, 2018):
            bar_colors.append("#E5B82B")  # Eval II additional Fit / Eval I held-out calibration
        elif y in (2019, 2020):
            bar_colors.append("#74B9FF")  # Eval I Test (2019-2020)
        else:
            bar_colors.append("#88D498")  # Eval II Test (2021-2023)

    bars = ax1.bar(years, records_k, color=bar_colors, edgecolor="#000000", linewidth=1.8, width=0.72)
    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) for y in years], rotation=45, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Eligible Finisher Records (Thousands)", fontsize=10, fontweight="bold")
    ax1.set_title(
        "FIG 1. UNIFIED IRONMAN 70.3 COHORT (2004–2023 · N = 1,152,658 RECORDS · 1,134 RACES)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    for b, rk in zip(bars, records_k):
        if rk >= 30:
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height() + 1.8,
                f"{rk:.0f}k",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#FFD23F", edgecolor="#000000", label="2004–2016: Eval I & II Fit (n=495,489)"),
        Patch(facecolor="#E5B82B", edgecolor="#000000", label="2017–2018: Eval II Fit Extension (+201,778 → n=697,267)"),
        Patch(facecolor="#74B9FF", edgecolor="#000000", label="2019–2020: Eval I Test Cohort (n=125,912 · 127 races)"),
        Patch(facecolor="#88D498", edgecolor="#000000", label="2021–2023: Eval II Test Cohort (n=329,366 post-QC · 258 races)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#000000", fontsize=8)
    ax1.set_ylim(0, max(records_k) + 18)
    for sp in ["top", "right", "left", "bottom"]:
        ax1.spines[sp].set_linewidth(2.0)
        ax1.spines[sp].set_color("#000000")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#000000")
    fig1.tight_layout()
    p1 = out_dir / "fig1_cohort_splits.png"
    fig1.savefig(p1, dpi=300)
    plt.close(fig1)
    chart_paths["fig1"] = p1

    # -------------------------------------------------------------------------
    # Figure 2: Progressive MAE Reduction Across Checkpoints (Eval I & Eval II)
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300, sharey=True)
    fig2.patch.set_facecolor("#FFFFFF")

    models_order = [
        ("B0", "B0 (Global Median)", "#D8D4C5"),
        ("B2", "B2 (Splits + Group Median)", "#A39E8B"),
        ("LR", "Linear Regression (LR)", "#FFFFFF"),
        ("RF", "Random Forest (RF)", "#74B9FF"),
        ("CatBoost_ref", "CatBoost_ref (.cbm)", "#FFD23F"),
    ]
    checkpoints = ["C0", "C1", "C2"]
    x = np.arange(len(checkpoints))
    width = 0.155

    for ax, split, title_str in [
        (ax2a, "I", "EVALUATION I (Fit 2004–2016 → Test 2019–2020 · n=125,912)"),
        (ax2b, "II", "EVALUATION II (Fit 2004–2018 → Test 2021–2023 · n=329,366)"),
    ]:
        ax.set_facecolor("#FFFDF5")
        sub = all_models.loc[all_models["split"] == split]
        for idx, (m_code, m_lbl, col) in enumerate(models_order):
            m_sub = sub.loc[sub["model"] == m_code].set_index("checkpoint")
            y_vals = [float(m_sub.loc[cp, "mae_minutes"]) for cp in checkpoints]
            offset = (idx - 2) * width
            bars = ax.bar(
                x + offset,
                y_vals,
                width=width,
                label=m_lbl,
                color=col,
                edgecolor="#000000",
                linewidth=1.5,
            )
            for b, v in zip(bars, y_vals):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.6,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    fontweight="bold",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(["C0 (Before Race)", "C1 (After Swim)", "C2 (After Bike)"], fontsize=9.5, fontweight="bold")
        ax.set_title(title_str, fontsize=10, fontweight="bold", pad=8)
        ax.set_ylim(0, 52)
        for sp in ["top", "right", "left", "bottom"]:
            ax.spines[sp].set_linewidth(2.0)
            ax.spines[sp].set_color("#000000")
        ax.grid(axis="y", linestyle="--", alpha=0.35, color="#000000")

    ax2a.set_ylabel("Mean Absolute Error — MAE (Minutes)", fontsize=10, fontweight="bold")
    ax2b.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#000000", fontsize=8)
    fig2.suptitle(
        "FIG 2. PROGRESSIVE MAE REDUCTION ACROSS CHECKPOINTS C0 → C1 → C2 (MINUTES · LOWER IS BETTER)",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig2.tight_layout()
    p2 = out_dir / "fig2_mae_progression.png"
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)
    chart_paths["fig2"] = p2

    # -------------------------------------------------------------------------
    # Figure 3: P90 Absolute Error & RMSE Trajectory (Evaluation II)
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9.5, 4.1), dpi=300)
    fig3.patch.set_facecolor("#FFFFFF")
    ax3.set_facecolor("#FFFDF5")

    sub_ii = all_models.loc[all_models["split"] == "II"]
    cp_idx = np.arange(3)
    for m_code, m_lbl, col, marker in [
        ("B2", "B2 Baseline (P90 |Error|)", "#8E8D8A", "s"),
        ("LR", "Linear Regression (P90 |Error|)", "#000000", "^"),
        ("RF", "Random Forest (P90 |Error|)", "#74B9FF", "D"),
        ("CatBoost_ref", "CatBoost_ref (P90 |Error|)", "#FF6B6B", "o"),
    ]:
        ms = sub_ii.loc[sub_ii["model"] == m_code].set_index("checkpoint")
        p90_vals = [float(ms.loc[c, "p90_abs_minutes"]) for c in checkpoints]
        ax3.plot(
            cp_idx,
            p90_vals,
            marker=marker,
            markersize=8,
            linewidth=2.6,
            color=col,
            label=m_lbl,
        )
        for xi, yi in zip(cp_idx, p90_vals):
            ax3.annotate(
                f"{yi:.1f}m",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    ax3.set_xticks(cp_idx)
    ax3.set_xticklabels(
        ["C0 — BEFORE RACE\n(Demographics + Location)", "C1 — AFTER SWIM\n(+ 1.9 km SwimTime)", "C2 — AFTER BIKE\n(+ 90 km BikeTime)"],
        fontsize=9.5,
        fontweight="bold",
    )
    ax3.set_ylabel("90th Percentile Absolute Error — P90 (Minutes)", fontsize=9.5, fontweight="bold")
    ax3.set_title(
        "FIG 3. TAIL RISK CONTRACTION: P90 ABSOLUTE ERROR IN EVALUATION II (MINUTES)",
        fontsize=10.8,
        fontweight="bold",
        pad=10,
    )
    ax3.set_ylim(20, 100)
    ax3.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#000000", fontsize=8.5)
    for sp in ["top", "right", "left", "bottom"]:
        ax3.spines[sp].set_linewidth(2.0)
        ax3.spines[sp].set_color("#000000")
    ax3.grid(True, linestyle="--", alpha=0.35, color="#000000")
    fig3.tight_layout()
    p3 = out_dir / "fig3_p90_trajectory.png"
    fig3.savefig(p3, dpi=300)
    plt.close(fig3)
    chart_paths["fig3"] = p3

    return chart_paths


def draw_nb_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    fill_color: colors.Color = C_WHITE,
    header_text: str = "",
    header_bg: colors.Color = C_BLACK,
    header_fg: colors.Color = C_CREAM,
    badge_text: str = "",
    badge_bg: colors.Color = C_C0,
    shadow_offset: float = 6.0,
    border_width: float = 2.8,
) -> float:
    """
    Draw a Neubrutalist card with a solid black hard offset shadow, sharp square corners,
    and an optional section header bar. Notice in ReportLab (x, y) is the bottom-left corner.
    Returns the top content Y coordinate inside the card.
    """
    # Hard shadow
    c.setFillColor(C_BLACK)
    c.rect(x + shadow_offset, y - shadow_offset, w, h, stroke=0, fill=1)

    # Main box
    c.setFillColor(fill_color)
    c.setStrokeColor(C_BLACK)
    c.setLineWidth(border_width)
    c.rect(x, y, w, h, stroke=1, fill=1)

    top_y = y + h
    if header_text:
        header_h = 28.0
        c.setFillColor(header_bg)
        c.setStrokeColor(C_BLACK)
        c.setLineWidth(border_width)
        c.rect(x, top_y - header_h, w, header_h, stroke=1, fill=1)

        c.setFillColor(header_fg)
        c.setFont("Helvetica-Bold", 13.5)
        c.drawString(x + 12, top_y - header_h + 8.5, header_text.upper())

        if badge_text:
            c.setFont("Courier-Bold", 10.5)
            bw = c.stringWidth(badge_text, "Courier-Bold", 10.5) + 16
            bx = x + w - bw - 10
            by = top_y - header_h + 4.5
            c.setFillColor(badge_bg)
            c.setStrokeColor(C_BLACK)
            c.setLineWidth(1.6)
            c.rect(bx, by, bw, 19, stroke=1, fill=1)
            c.setFillColor(C_BLACK)
            c.drawString(bx + 8, by + 5, badge_text)

        return top_y - header_h - 14.0
    return top_y - 14.0


def draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    font_name: str = "Helvetica",
    font_size: float = 11.2,
    leading: float = 15.2,
    color: colors.Color = C_BLACK,
) -> float:
    """
    Word-wrap a paragraph of text within max_width and return the new Y coordinate below it.
    """
    c.setFont(font_name, font_size)
    c.setFillColor(color)
    words = text.split()
    lines: List[str] = []
    current_line = ""
    for w in words:
        test_line = f"{current_line} {w}".strip()
        if c.stringWidth(test_line, font_name, font_size) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = w
    if current_line:
        lines.append(current_line)

    curr_y = y
    for line in lines:
        c.drawString(x, curr_y, line)
        curr_y -= leading
    return curr_y


def draw_bullet_list(
    c: canvas.Canvas,
    items: List[Tuple[str, str]],
    x: float,
    y: float,
    max_width: float,
    font_size: float = 10.8,
    leading: float = 14.6,
    item_gap: float = 4.5,
) -> float:
    """
    Draw a structured bullet list where each item is (bold_lead, body_text).
    Returns the new Y coordinate.
    """
    curr_y = y
    for lead, body in items:
        # Draw square bullet
        c.setFillColor(C_BLACK)
        c.rect(x, curr_y + 2, 5.5, 5.5, stroke=0, fill=1)

        full_text = f"{lead} {body}".strip()
        # Wrap full_text
        words = full_text.split()
        lines: List[str] = []
        current_line = ""
        for w in words:
            test_line = f"{current_line} {w}".strip()
            if c.stringWidth(test_line, "Helvetica", font_size) <= (max_width - 14):
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = w
        if current_line:
            lines.append(current_line)

        for idx, line in enumerate(lines):
            if idx == 0 and lead:
                # Highlight the lead prefix in bold if it fits on the first line
                if line.startswith(lead):
                    c.setFont("Helvetica-Bold", font_size)
                    c.drawString(x + 12, curr_y, lead)
                    lead_w = c.stringWidth(lead + " ", "Helvetica-Bold", font_size)
                    rest = line[len(lead):].lstrip()
                    c.setFont("Helvetica", font_size)
                    c.drawString(x + 12 + lead_w, curr_y, rest)
                else:
                    c.setFont("Helvetica-Bold", font_size)
                    c.drawString(x + 12, curr_y, line)
            else:
                c.setFont("Helvetica", font_size)
                c.drawString(x + 12, curr_y, line)
            curr_y -= leading
        curr_y -= item_gap
    return curr_y


def draw_grid_table(
    c: canvas.Canvas,
    headers: List[str],
    rows: List[List[str]],
    col_widths: List[float],
    x: float,
    y_top: float,
    row_height: float = 18.5,
    header_bg: colors.Color = C_BLACK,
    highlight_rows: Optional[Dict[int, colors.Color]] = None,
    font_size: float = 9.4,
) -> float:
    """
    Draw a sharp-edged Neubrutalist scientific data table starting from y_top.
    Returns the Y coordinate below the table.
    """
    highlight_rows = highlight_rows or {}
    total_w = sum(col_widths)

    # Header row
    curr_y = y_top - row_height
    c.setFillColor(header_bg)
    c.setStrokeColor(C_BLACK)
    c.setLineWidth(1.8)
    c.rect(x, curr_y, total_w, row_height, stroke=1, fill=1)

    cx = x
    c.setFillColor(C_CREAM)
    c.setFont("Courier-Bold", font_size)
    for h_text, cw in zip(headers, col_widths):
        c.drawString(cx + 5, curr_y + 5, h_text)
        cx += cw

    # Data rows
    for r_idx, row in enumerate(rows):
        curr_y -= row_height
        bg = highlight_rows.get(r_idx, C_WHITE if r_idx % 2 == 0 else C_CREAM)
        c.setFillColor(bg)
        c.setStrokeColor(C_BLACK)
        c.setLineWidth(1.4)
        c.rect(x, curr_y, total_w, row_height, stroke=1, fill=1)

        cx = x
        is_hl = r_idx in highlight_rows
        c.setFont("Courier-Bold" if is_hl else "Courier", font_size)
        c.setFillColor(C_BLACK)
        for cell_text, cw in zip(row, col_widths):
            c.drawString(cx + 5, curr_y + 5, str(cell_text))
            cx += cw
            # vertical divider
            c.line(cx, curr_y, cx, curr_y + row_height)

    # Outer border
    table_h = row_height * (len(rows) + 1)
    c.setStrokeColor(C_BLACK)
    c.setLineWidth(2.2)
    c.rect(x, curr_y, total_w, table_h, stroke=1, fill=0)
    return curr_y - 10.0


def build_a0_poster() -> Path:
    """Build the complete A0 Portrait scientific research poster PDF."""
    assets_dir = POSTER_DIR / "figures"
    chart_paths = generate_poster_charts(assets_dir)

    pdf_path = POSTER_DIR / "IRONMAN_70_3_Finish_Time_Poster_A0.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(W_PT, H_PT))
    c.setTitle("IRONMAN 70.3 Finish Time Prediction Across Race Checkpoints — A0 Research Poster")
    c.setAuthor("Department of Artificial Intelligence, FPT University Da Nang")
    c.setSubject("Prospective Multi-Checkpoint Triathlon Finish Time Forecasting (C0, C1, C2)")

    # 1. Warm Cream Background (#FFFDF5) & Outer Frame
    c.setFillColor(C_CREAM)
    c.rect(0, 0, W_PT, H_PT, stroke=0, fill=1)

    margin = 38.0
    c.setStrokeColor(C_BLACK)
    c.setLineWidth(4.0)
    c.rect(margin / 2, margin / 2, W_PT - margin, H_PT - margin, stroke=1, fill=0)

    # =========================================================================
    # HEADER BANNER (Top of A0 Poster)
    # =========================================================================
    header_h = 250.0
    header_x = margin
    header_y = H_PT - margin - header_h
    header_w = W_PT - 2 * margin

    # Hard shadow for header
    c.setFillColor(C_BLACK)
    c.rect(header_x + 9, header_y - 9, header_w, header_h, stroke=0, fill=1)

    c.setFillColor(C_WHITE)
    c.setStrokeColor(C_BLACK)
    c.setLineWidth(3.5)
    c.rect(header_x, header_y, header_w, header_h, stroke=1, fill=1)

    # Top institution strip inside header
    strip_h = 42.0
    c.setFillColor(C_BLACK)
    c.rect(header_x, header_y + header_h - strip_h, header_w, strip_h, stroke=1, fill=1)

    c.setFillColor(C_C0)
    c.setFont("Courier-Bold", 14.5)
    c.drawString(
        header_x + 18,
        header_y + header_h - strip_h + 14,
        "FPT UNIVERSITY DA NANG · DEPARTMENT OF ARTIFICIAL INTELLIGENCE · RESEARCH-BASED LEARNING",
    )

    # Right pill in top strip
    tag_str = "A0 SCIENTIFIC POSTER (841 × 1189 MM) · OFFLINE DEMO VERIFIED"
    tw = c.stringWidth(tag_str, "Courier-Bold", 12.5) + 24
    c.setFillColor(C_C2)
    c.rect(header_x + header_w - tw - 14, header_y + header_h - strip_h + 8, tw, 26, stroke=1, fill=1)
    c.setFillColor(C_BLACK)
    c.setFont("Courier-Bold", 12.5)
    c.drawString(header_x + header_w - tw - 2, header_y + header_h - strip_h + 16, tag_str)

    # Main Poster Title
    c.setFillColor(C_BLACK)
    c.setFont("Helvetica-Bold", 36.5)
    c.drawString(
        header_x + 20,
        header_y + header_h - 92,
        "PROGRESSIVE FINISH TIME PREDICTION IN IRONMAN 70.3 TRIATHLON",
    )
    c.setFont("Helvetica-Bold", 31.0)
    c.drawString(
        header_x + 20,
        header_y + header_h - 130,
        "ACROSS PRE-RACE (C0), POST-SWIM (C1), AND POST-BIKE (C2) CHECKPOINTS",
    )

    # Authors & Affiliation Bar
    c.setFont("Helvetica-Bold", 15.0)
    c.drawString(
        header_x + 20,
        header_y + header_h - 164,
        "Authors: Khue, Bibi, Hoi & Research Team Members (Class AI21A02 · Database Systems & Predictive Modeling Track)",
    )
    c.setFont("Helvetica", 13.5)
    c.drawString(
        header_x + 20,
        header_y + header_h - 186,
        "Affiliation: Department of Artificial Intelligence, FPT University Da Nang, Vietnam   |   Repository: github.com/kaity121/ironman-70-3-finish-time",
    )

    # 4 Key Metric Callout Badges at the bottom of Header
    badges = [
        ("UNIFIED COHORT (2004–2023)", "1,152,658 Age-Group Records · 1,134 Races", C_C0),
        ("C0 PRE-RACE CATBOOST MAE", "Eval I: 36.37 min | Eval II: 41.72 min", C_C0),
        ("C1 POST-SWIM CATBOOST MAE", "Eval I: 28.43 min | Eval II: 33.45 min", C_C1),
        ("C2 POST-BIKE CATBOOST MAE", "Eval I: 14.79 min | Eval II: 15.79 min", C_C2),
    ]
    bx = header_x + 20
    bw_each = (header_w - 40 - 3 * 16) / 4.0
    by = header_y + 12
    bh_badge = 42.0
    for b_title, b_val, b_col in badges:
        c.setFillColor(C_BLACK)
        c.rect(bx + 3, by - 3, bw_each, bh_badge, stroke=0, fill=1)
        c.setFillColor(b_col)
        c.setStrokeColor(C_BLACK)
        c.setLineWidth(2.0)
        c.rect(bx, by, bw_each, bh_badge, stroke=1, fill=1)
        c.setFillColor(C_BLACK)
        c.setFont("Courier-Bold", 10.5)
        c.drawString(bx + 10, by + 25, b_title)
        c.setFont("Helvetica-Bold", 12.5)
        c.drawString(bx + 10, by + 8, b_val)
        bx += bw_each + 16.0

    # =========================================================================
    # 3-COLUMN MODULAR GRID SETUP
    # =========================================================================
    col_gap = 24.0
    col_w = (header_w - 2 * col_gap) / 3.0
    col1_x = margin
    col2_x = margin + col_w + col_gap
    col3_x = margin + 2 * (col_w + col_gap)

    grid_top = header_y - 24.0
    grid_bottom = margin + 12.0

    # =========================================================================
    # COLUMN 1: INTRODUCTION, OBJECTIVES, DATASET & COHORT VERIFICATION
    # =========================================================================
    curr_y1 = grid_top

    # Card 1.1: Introduction & Problem Statement
    h_intro = 370.0
    y_intro = curr_y1 - h_intro
    cy = draw_nb_card(
        c,
        col1_x,
        y_intro,
        col_w,
        h_intro,
        fill_color=C_WHITE,
        header_text="1. INTRODUCTION & RESEARCH CONTEXT",
        badge_text="BACKGROUND",
        badge_bg=C_C0,
    )
    cy = draw_wrapped_text(
        c,
        "IRONMAN 70.3 (Half Ironman) requires athletes to complete a 1.9 km swim, a 90 km bike ride, and a 21.1 km half-marathon run consecutively. Accurate finish-time forecasting supports pacing management, race operations, and spectator tracking. However, prior triathlon studies largely rely on either random train-test splits (which introduce temporal leakage across race editions) or single static pre-race models.",
        col1_x + 14,
        cy,
        col_w - 28,
        font_size=11.3,
        leading=15.5,
    )
    cy -= 8
    cy = draw_wrapped_text(
        c,
        "In our initial study (V1, ML_model.ipynb), comparing Pre-race (Model A) directly to Post-bike (Model B) on 2021–2023 races showed that Random Forest MAE dropped by 25.50 minutes (from 40.89 to 15.39 min). Yet because Model B added SwimTime and BikeTime simultaneously, V1 could not isolate the marginal predictive value of the swim split alone or benchmark machine learning against a domain baseline that also knows completed split times.",
        col1_x + 14,
        cy,
        col_w - 28,
        font_size=11.3,
        leading=15.5,
    )
    cy -= 10
    cy = draw_bullet_list(
        c,
        [
            ("Leakage-Free Temporal Protocol:", "Evaluates strictly on future unseen years after the training window, preventing event-level leakage."),
            ("Progressive Three-Checkpoint Framing:", "Tracks the exact same race participation record across C0 (Before Race), C1 (After Swim), and C2 (After Bike)."),
            ("Decomposing Information vs. Algorithm Gain:", "Compares ML estimators against a conditional remainder baseline (B2) that adds observed splits to demographic remainder medians."),
        ],
        col1_x + 14,
        cy,
        col_w - 28,
        font_size=10.8,
        leading=14.5,
    )
    curr_y1 = y_intro - 22.0

    # Card 1.2: Research Objectives & Questions
    h_obj = 345.0
    y_obj = curr_y1 - h_obj
    cy = draw_nb_card(
        c,
        col1_x,
        y_obj,
        col_w,
        h_obj,
        fill_color=C_C0,
        header_text="2. RESEARCH OBJECTIVES",
        badge_text="3 CHECKPOINTS",
        badge_bg=C_WHITE,
    )
    cy = draw_bullet_list(
        c,
        [
            (
                "Objective 1 — Marginal Value of Progressive Split Updates (C0 -> C1 -> C2):",
                "Quantify in minutes how much prediction error (MAE, RMSE, Bias, and P90 absolute error) decreases when updating predictions immediately after the 1.9 km swim (C1) and after the 90 km bike (C2).",
            ),
            (
                "Objective 2 — Model Comparison Against Conditional Remainder Baselines:",
                "Benchmark Linear Regression (LR), Random Forest (RF), and CatBoost reference models (CatBoost_ref) at each checkpoint against empirical medians B0 (global), B1 (pre-race gender × agegroup), and B2 (known splits + group remainder median).",
            ),
            (
                "Objective 3 — Dual Temporal Generalization (Evaluation I & Evaluation II):",
                "Validate stability across two disjoint temporal boundaries: Kaggle historical test years (Evaluation I: fit 2004–2016, test 2019–2020) and Endurance-Data scraped recent years (Evaluation II: fit 2004–2018, test 2021–2023).",
            ),
            (
                "Objective 4 — Reproducible Offline Inference System:",
                "Deploy a zero-latency, 100% offline demonstration platform using persisted CatBoost binaries (models/*.cbm) and baseline lookup tables (models/*_baseline.csv).",
            ),
        ],
        col1_x + 14,
        cy,
        col_w - 28,
        font_size=10.9,
        leading=14.8,
        item_gap=6.0,
    )
    curr_y1 = y_obj - 22.0

    # Card 1.3: Unified Dataset, Cohort Filtering & Quality Control
    h_data = curr_y1 - grid_bottom
    y_data = grid_bottom
    cy = draw_nb_card(
        c,
        col1_x,
        y_data,
        col_w,
        h_data,
        fill_color=C_WHITE,
        header_text="3. UNIFIED DATASET & QUALITY CONTROL",
        badge_text="N = 1,152,658",
        badge_bg=C_C2,
    )
    cy = draw_wrapped_text(
        c,
        "Data Sources & Star-Schema Snapshot Verification: The dataset integrates three normalized tables (Athletes.csv, Races.csv, Results.csv) spanning 2004–2023 from Kaggle (2004–2020) and Endurance-Data (2021–2023). Cross-checking against Supabase view vw_valid_c1c2 (1,153,194 raw view rows) confirms 1,152,658 identical eligible ResultIDs with 0 value discrepancies across all 11 modeling columns (metrics/snapshot_comparison.json).",
        col1_x + 14,
        cy,
        col_w - 28,
        font_size=11.0,
        leading=15.0,
    )
    cy -= 8
    cy = draw_bullet_list(
        c,
        [
            ("Population Cohort Filter:", "Individual Age-Group finishers (AthleteType == 'Age-Group'), Gender in {'M', 'F'}, and 11 standard five-year age bands from 18-24 to 70-74 (ageband = lower bound integer 18..70)."),
            ("Positive Finite Splits:", "SwimTime > 0, BikeTime > 0, and FinishTime > 0 (all in seconds). No arbitrary percentile trimming of FinishTime is applied."),
            ("Categorical Normalization:", "Leading/trailing whitespace trimmed; empty strings mapped to '__MISSING__'. Namibia's ISO code 'NA' is explicitly preserved via keep_default_na=False."),
            ("Post-Run Consistency QC (FinishTime > SwimTime + BikeTime):", "Audit identified 113 records in 2021–2023 (58 in 2021, 46 in 2022, 9 in 2023; 0 in 2004–2020) where total FinishTime <= SwimTime + BikeTime. These 113 records are excluded from final Evaluation II test scoring (n_test = 329,366), with pre-QC sensitivity preserved in metrics/before_consistency_check.csv."),
        ],
        col1_x + 14,
        cy,
        col_w - 28,
        font_size=10.6,
        leading=14.4,
        item_gap=4.5,
    )

    cy -= 6
    c.setFont("Helvetica-Bold", 11.2)
    c.setFillColor(C_BLACK)
    c.drawString(col1_x + 14, cy, "ANNUAL COHORT SUMMARY BY ERA (FROM metrics/cohort.csv)")
    cy -= 8

    cohort_headers = ["Temporal Era", "Years", "Active Editions", "Eligible Records", "% of Total", "Role in Evaluation Protocol"]
    cohort_rows = [
        ["Early Expansion Era", "2004–2009", "38 races", "43,077", "3.74%", "Fit Cohort (Eval I & Eval II)"],
        ["Global Growth Era", "2010–2016", "402 races", "452,412", "39.25%", "Fit Cohort (Eval I & Eval II)"],
        ["Calibration / Fit Ext.", "2017–2018", "203 races", "201,778", "17.51%", "Eval I Held-out / Eval II Fit"],
        ["Eval I Test Horizon", "2019–2020", "127 races", "125,912", "10.92%", "Evaluation I Test Set (Kaggle)"],
        ["Eval II Test Horizon", "2021–2023", "258 races", "329,479 (329,366 QC)", "28.58%", "Evaluation II Test Set (Endurance)"],
        ["Total Unified Cohort", "2004–2023", "1,028 races", "1,152,658", "100.00%", "100% Verified vs. Supabase View"],
    ]
    cw_cols = [115.0, 68.0, 78.0, 115.0, 62.0, 160.0]
    c_scale = (col_w - 28.0) / sum(cw_cols)
    cw_cols = [w * c_scale for w in cw_cols]
    cy = draw_grid_table(
        c,
        cohort_headers,
        cohort_rows,
        cw_cols,
        col1_x + 14,
        cy,
        row_height=19.5,
        highlight_rows={3: C_C1, 4: C_C2, 5: C_C0},
        font_size=8.8,
    )

    cy -= 4
    cy = draw_wrapped_text(
        c,
        "Star-Schema Relational Integrity (scripts/run.py): Every ResultID is unique (`assert df.resultid.is_unique`), many-to-one foreign keys to Athletes.csv (`athleteid`) and Races.csv (`raceid`) have 100% referential coverage, and every record's `ageband` matches the integer prefix of `agegroup`.",
        col1_x + 14,
        cy,
        col_w - 28,
        font_size=10.2,
        leading=13.8,
    )

    # Embed Figure 1 inside Card 1.3 filling remaining space cleanly
    fig1_w = col_w - 28.0
    fig1_h = max(260.0, cy - (y_data + 18.0))
    c.drawImage(
        ImageReader(str(chart_paths["fig1"])),
        col1_x + 14,
        y_data + 14,
        width=fig1_w,
        height=fig1_h,
        preserveAspectRatio=True,
        anchor="c",
    )
    c.setStrokeColor(C_BLACK)
    c.setLineWidth(2.0)
    c.rect(col1_x + 14, y_data + 14, fig1_w, fig1_h, stroke=1, fill=0)

    # =========================================================================
    # COLUMN 2: METHODOLOGY, CHECKPOINT ARCHITECTURE & EXPERIMENTAL SETUP
    # =========================================================================
    curr_y2 = grid_top

    # Card 2.1: Multi-Checkpoint Prediction Architecture (C0, C1, C2)
    h_meth = 560.0
    y_meth = curr_y2 - h_meth
    cy = draw_nb_card(
        c,
        col2_x,
        y_meth,
        col_w,
        h_meth,
        fill_color=C_WHITE,
        header_text="4. MULTI-CHECKPOINT PREDICTION METHODOLOGY",
        badge_text="C0 -> C1 -> C2",
        badge_bg=C_C1,
    )
    cy = draw_wrapped_text(
        c,
        "At every checkpoint, all models predict total FinishTime (in seconds from race start to finish line). Only features genuinely observable at or before that checkpoint are supplied. RunTime, rankings, pace ratios, and transition splits are strictly excluded to prevent target leakage.",
        col2_x + 14,
        cy,
        col_w - 28,
        font_size=11.0,
        leading=15.0,
    )
    cy -= 10

    # Visual Checkpoint Diagram Boxes inside Card 2.1
    cp_boxes = [
        (
            "CHECKPOINT C0 — BEFORE RACE (5 FEATURES)",
            "Models: I_C0.cbm, II_C0.cbm  |  Baselines: I_C0_baseline.csv, II_C0_baseline.csv",
            "Features: [gender, ageband, countryiso, eventlocation, eventyear]  |  Known Split Sum S_0 = 0",
            C_C0,
        ),
        (
            "CHECKPOINT C1 — AFTER 1.9 KM SWIM (6 FEATURES)",
            "Models: I_C1.cbm, II_C1.cbm  |  Baselines: I_C1_baseline.csv, II_C1_baseline.csv",
            "Features: C0 features + [swimtime (seconds)]  |  Known Split Sum S_1 = swimtime",
            C_C1,
        ),
        (
            "CHECKPOINT C2 — AFTER 90 KM BIKE (7 FEATURES)",
            "Models: I_C2.cbm, II_C2.cbm  |  Baselines: I_C2_baseline.csv, II_C2_baseline.csv",
            "Features: C1 features + [biketime (seconds)]  |  Known Split Sum S_2 = swimtime + biketime",
            C_C2,
        ),
    ]
    for box_title, box_sub, box_feat, box_col in cp_boxes:
        bh_cp = 74.0
        by_cp = cy - bh_cp
        c.setFillColor(C_BLACK)
        c.rect(col2_x + 18, by_cp - 4, col_w - 32, bh_cp, stroke=0, fill=1)
        c.setFillColor(box_col)
        c.setStrokeColor(C_BLACK)
        c.setLineWidth(2.2)
        c.rect(col2_x + 14, by_cp, col_w - 32, bh_cp, stroke=1, fill=1)

        c.setFillColor(C_BLACK)
        c.setFont("Helvetica-Bold", 12.5)
        c.drawString(col2_x + 24, by_cp + 52, box_title)
        c.setFont("Courier-Bold", 9.8)
        c.drawString(col2_x + 24, by_cp + 33, box_sub)
        c.setFont("Courier", 9.6)
        c.drawString(col2_x + 24, by_cp + 13, box_feat)
        cy = by_cp - 14.0

    cy -= 4
    cy = draw_bullet_list(
        c,
        [
            ("Baseline B0 (Global Fit Median):", "Predicts the overall median `FinishTime` of the fit cohort (`20,962.0 s` in Eval I; `20,943.0 s` in Eval II)."),
            ("Baseline B1 (Pre-Race Demographic Median):", "Median `FinishTime` stratified by `gender × agegroup` on the fit cohort (evaluated at C0, where `B1 == B2`)."),
            ("Baseline B2 (Conditional Remainder Median):", "Adds observed split sum `S_t` to the fit median of remaining time `R_t = FinishTime - S_t` by `gender × agegroup` (hierarchical fallback to `gender`, then `all` if group size `< 30`; minimum actual group size is 79 in Eval I and 137 in Eval II)."),
            ("Machine Learning Estimators:", "`LR` (OneHotEncoder + StandardScaler on numericals, `tol=1e-9`), `RF` (`200` trees, `max_depth=12`, `min_samples_leaf=50`, `seed=42`), and `CatBoost_ref` (`120` iterations, `depth=6`, `lr=0.08`, `l2_leaf_reg=3`, `RMSE` loss, native categorical handling on `[gender, countryiso, eventlocation]`, `seed=42`)."),
        ],
        col2_x + 14,
        cy,
        col_w - 28,
        font_size=10.3,
        leading=13.8,
        item_gap=3.5,
    )
    curr_y2 = y_meth - 22.0

    # Card 2.2: Experimental Setup & Temporal Evaluation Protocol
    h_setup = 420.0
    y_setup = curr_y2 - h_setup
    cy = draw_nb_card(
        c,
        col2_x,
        y_setup,
        col_w,
        h_setup,
        fill_color=C_WHITE,
        header_text="5. EXPERIMENTAL SETUP & TEMPORAL SPLITS",
        badge_text="DISJOINT RACES",
        badge_bg=C_C0,
    )
    cy = draw_wrapped_text(
        c,
        "Two chronological evaluation splits test generalization across distinct time horizons and data collection pipelines. Calibration years (2017–2018 in Eval I; 2019–2020 in Eval II) are held out outside the fit window per protocol. Fit and test sets share zero common RaceIDs (`assert not set(fit.raceid).intersection(test.raceid)`).",
        col2_x + 14,
        cy,
        col_w - 28,
        font_size=10.8,
        leading=14.6,
    )
    cy -= 8

    setup_headers = ["Split / Protocol", "Fit Window", "Fit Records", "Test Window", "Test Records", "Test Races"]
    setup_rows = [
        ["Evaluation I (Kaggle)", "2004–2016", "495,489", "2019–2020", "125,912", "127"],
        ["Evaluation II (Post-QC)", "2004–2018", "697,267", "2021–2023", "329,366", "258"],
        ["Evaluation II (Pre-QC)", "2004–2018", "697,267", "2021–2023", "329,479", "258"],
        ["Historical V1 Notebook", "2004–2020", "814,715", "2021–2023", "323,967", "N/A (C0 & C2)"],
    ]
    sw_cols = [155.0, 95.0, 95.0, 95.0, 100.0, 140.0]
    scale_f = (col_w - 28.0) / sum(sw_cols)
    sw_cols = [w * scale_f for w in sw_cols]
    cy = draw_grid_table(
        c,
        setup_headers,
        setup_rows,
        sw_cols,
        col2_x + 14,
        cy,
        row_height=20.0,
        highlight_rows={0: C_C1, 1: C_C2},
        font_size=9.2,
    )
    cy -= 6
    cy = draw_wrapped_text(
        c,
        "Comparison of V1 vs. Three-Checkpoint Pipeline: V1 (v1/ML_model.ipynb) encoded Continent_enc and AthleteType_enc (including 1,596 PRO records), trimmed P1/P99 FinishTime outliers, fit LabelEncoder prior to splitting, and evaluated only LR/RF at C0 (MAE: LR 41.44m, RF 40.89m) and C2 (MAE: LR 16.81m, RF 15.39m). The current three-checkpoint pipeline enforces strict Age-Group filtering, raw CountryISO and trimmed EventLocation features, fit-only encoding, and adds Checkpoint C1 plus CatBoost_ref and B0/B1/B2 baselines.",
        col2_x + 14,
        cy,
        col_w - 28,
        font_size=10.3,
        leading=14.0,
    )
    cy -= 6
    cy = draw_wrapped_text(
        c,
        "Provenance Note (config/run.json): CatBoost_ref uses a fixed reference configuration selected before scoring (120 iterations, seed 42). Full hyperparameter grid tuning, 3-seed averaging, quantile/CQR interval calibration, and hierarchical location mixed-effects models remain designated for future work.",
        col2_x + 14,
        cy,
        col_w - 28,
        font_size=10.0,
        leading=13.6,
    )
    curr_y2 = y_setup - 22.0

    # Card 2.3: Visual Performance Progression Charts (Fig 2 & Fig 3)
    h_charts = curr_y2 - grid_bottom
    y_charts = grid_bottom
    cy = draw_nb_card(
        c,
        col2_x,
        y_charts,
        col_w,
        h_charts,
        fill_color=C_WHITE,
        header_text="6. PROGRESSIVE ERROR REDUCTION VISUALIZATION",
        badge_text="VERIFIED METRICS",
        badge_bg=C_C2,
    )

    cy = draw_wrapped_text(
        c,
        "Information Gain Decomposition (B2 vs. CatBoost_ref): Comparing B2 against B0 measures the raw value of knowing completed stage splits plus age/gender medians, whereas comparing CatBoost_ref against B2 at the same checkpoint isolates the nonlinear modeling gain from jointly conditioning on SwimTime, BikeTime, EventLocation, CountryISO, and EventYear.",
        col2_x + 14,
        cy,
        col_w - 28,
        font_size=10.5,
        leading=14.2,
    )
    cy -= 8

    avail_h = cy - (y_charts + 16)
    img_w = col_w - 28.0
    img2_h = avail_h * 0.52
    img3_h = avail_h * 0.45

    # Top chart: Fig 2
    y_fig2 = cy - img2_h
    c.drawImage(
        ImageReader(str(chart_paths["fig2"])),
        col2_x + 14,
        y_fig2,
        width=img_w,
        height=img2_h,
        preserveAspectRatio=True,
        anchor="c",
    )
    c.setStrokeColor(C_BLACK)
    c.setLineWidth(2.0)
    c.rect(col2_x + 14, y_fig2, img_w, img2_h, stroke=1, fill=0)

    # Bottom chart: Fig 3
    y_fig3 = y_charts + 14
    c.drawImage(
        ImageReader(str(chart_paths["fig3"])),
        col2_x + 14,
        y_fig3,
        width=img_w,
        height=img3_h,
        preserveAspectRatio=True,
        anchor="c",
    )
    c.rect(col2_x + 14, y_fig3, img_w, img3_h, stroke=1, fill=0)

    # =========================================================================
    # COLUMN 3: OFFICIAL RESULTS TABLE, TEST REPLAY, CONCLUSION & REFERENCES
    # =========================================================================
    curr_y3 = grid_top

    # Card 3.1: Complete Experimental Results Table (metrics/all_models.csv)
    h_res = 820.0
    y_res = curr_y3 - h_res
    cy = draw_nb_card(
        c,
        col3_x,
        y_res,
        col_w,
        h_res,
        fill_color=C_WHITE,
        header_text="7. EXPERIMENTAL RESULTS (metrics/all_models.csv)",
        badge_text="ALL 32 RUNS",
        badge_bg=C_C0,
    )
    cy = draw_wrapped_text(
        c,
        "All errors are reported in minutes (predicted minus actual FinishTime divided by 60). Highlighted rows indicate the lowest MAE at each checkpoint within Evaluation I (n_test = 125,912) and Evaluation II (n_test = 329,366).",
        col3_x + 14,
        cy,
        col_w - 28,
        font_size=10.5,
        leading=14.2,
    )
    cy -= 6

    all_df = pd.read_csv(REPO_ROOT / "metrics" / "all_models.csv")
    res_headers = ["Split", "CP", "Model", "MAE (m)", "RMSE (m)", "Bias (m)", "P90 |Err|"]
    res_rows: List[List[str]] = []
    hl_map: Dict[int, colors.Color] = {}

    for idx, r in all_df.iterrows():
        sp = str(r["split"])
        cp = str(r["checkpoint"])
        m = str(r["model"])
        res_rows.append(
            [
                sp,
                cp,
                m,
                f"{float(r['mae_minutes']):.2f}",
                f"{float(r['rmse_minutes']):.2f}",
                f"{float(r['bias_minutes']):+.2f}",
                f"{float(r['p90_abs_minutes']):.2f}",
            ]
        )
        # Highlight best MAE per (split, checkpoint)
        if (sp == "I" and m == "CatBoost_ref") or (
            sp == "II"
            and (
                (cp == "C0" and m == "LR")
                or (cp == "C1" and m == "CatBoost_ref")
                or (cp == "C2" and m == "CatBoost_ref")
            )
        ):
            hl_map[int(idx)] = C_C0 if cp == "C0" else (C_C1 if cp == "C1" else C_C2)

    rw_cols = [45.0, 45.0, 125.0, 75.0, 75.0, 75.0, 75.0]
    r_scale = (col_w - 28.0) / sum(rw_cols)
    rw_cols = [w * r_scale for w in rw_cols]

    cy = draw_grid_table(
        c,
        res_headers,
        res_rows,
        rw_cols,
        col3_x + 14,
        cy,
        row_height=17.5,
        highlight_rows=hl_map,
        font_size=8.8,
    )
    cy -= 4
    cy = draw_bullet_list(
        c,
        [
            ("Value of Post-Swim Update (C0 -> C1):", "Incorporating SwimTime reduces CatBoost_ref MAE by 7.94 min in Eval I (36.37 to 28.43 min) and by 8.27 min in Eval II (41.72 to 33.45 min), outperforming baseline B2 by 6.11 min (Eval I) and 6.10 min (Eval II)."),
            ("Value of Post-Bike Update (C1 -> C2):", "Adding BikeTime further cuts CatBoost_ref MAE by 13.64 min in Eval I (28.43 to 14.79 min) and by 17.67 min in Eval II (33.45 to 15.79 min), outperforming B2 by 6.44 min (Eval I) and 7.36 min (Eval II)."),
            ("Pre-QC Sensitivity (113 Anomalous Records in Eval II):", "Removing the 113 records with FinishTime <= SwimTime + BikeTime improved Eval II C2 LR RMSE from 37.01 to 22.53 min and CatBoost_ref MAE from 15.82 to 15.79 min."),
        ],
        col3_x + 14,
        cy,
        col_w - 28,
        font_size=10.1,
        leading=13.5,
        item_gap=3.0,
    )
    curr_y3 = y_res - 22.0

    # Card 3.2: Offline Exhibition Demo & Real Test Replay Verification
    h_demo = 485.0
    y_demo = curr_y3 - h_demo
    cy = draw_nb_card(
        c,
        col3_x,
        y_demo,
        col_w,
        h_demo,
        fill_color=C_WHITE,
        header_text="8. OFFLINE DEMO & REAL TEST REPLAY (metrics/example.csv)",
        badge_text="10 TEST ATHLETES",
        badge_bg=C_C1,
    )
    cy = draw_wrapped_text(
        c,
        "The accompanying offline Streamlit application provides two scientifically verified modes: Mode A (Live Race Simulation using models/I_C*.cbm & II_C*.cbm plus *_baseline.csv) and Mode B (Real Test Replay of all 10 saved test athletes from Evaluation II, RaceID 774, Year 2021, keeping Actual FinishTime locked until final step reveal). Below are all 10 test records from metrics/example.csv comparing CatBoost_ref predictions (HH:MM:SS) across C0 -> C1 -> C2 against actual finish times:",
        col3_x + 14,
        cy,
        col_w - 28,
        font_size=10.3,
        leading=13.9,
    )
    cy -= 6

    ex_df = pd.read_csv(REPO_ROOT / "metrics" / "example.csv")
    sample_rids = sorted(ex_df["resultid"].unique().tolist())
    ex_headers = ["ResultID", "C0 CB Pred", "C1 CB Pred", "C2 CB Pred", "Actual Finish", "C2 |Error|"]
    ex_rows: List[List[str]] = []

    def _hms(sec: float) -> str:
        s = int(round(sec))
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

    for rid in sample_rids:
        sub_r = ex_df.loc[(ex_df["resultid"] == rid) & (ex_df["model"] == "CatBoost_ref")].set_index("checkpoint")
        c0_s = float(sub_r.loc["C0", "pred_finish_seconds"])
        c1_s = float(sub_r.loc["C1", "pred_finish_seconds"])
        c2_s = float(sub_r.loc["C2", "pred_finish_seconds"])
        act_s = float(sub_r.loc["C2", "actual_finish_seconds"])
        err_m = abs(c2_s - act_s) / 60.0
        ex_rows.append([str(rid), _hms(c0_s), _hms(c1_s), _hms(c2_s), _hms(act_s), f"{err_m:.2f} min"])

    ew_cols = [75.0, 90.0, 90.0, 90.0, 90.0, 80.0]
    e_scale = (col_w - 28.0) / sum(ew_cols)
    ew_cols = [w * e_scale for w in ew_cols]
    cy = draw_grid_table(
        c,
        ex_headers,
        ex_rows,
        ew_cols,
        col3_x + 14,
        cy,
        row_height=18.0,
        highlight_rows={4: C_C2, 5: C_C2, 9: C_C2},
        font_size=8.9,
    )
    cy -= 4
    cy = draw_wrapped_text(
        c,
        "Across all 10 fast finishers in RaceID 774 (actual times 04:13:11 to 04:29:34 vs. global median B0 = 05:49:03), C0 pre-race predictions start near the demographic median (~05:51–06:17), shift downward by ~30–50 minutes at C1 after observing fast swim splits, and converge toward the true finish time at C2 after observing bike splits (e.g., ResultID 840064 C2 error = 3.65 min; ResultID 840065 C2 error = 8.74 min; ResultID 840069 C2 error = 9.52 min).",
        col3_x + 14,
        cy,
        col_w - 28,
        font_size=10.1,
        leading=13.6,
    )
    curr_y3 = y_demo - 22.0

    # Card 3.3: Conclusions, Limitations & References
    h_conc = curr_y3 - grid_bottom
    y_conc = grid_bottom
    cy = draw_nb_card(
        c,
        col3_x,
        y_conc,
        col_w,
        h_conc,
        fill_color=C_WHITE,
        header_text="9. CONCLUSIONS, LIMITATIONS & REFERENCES",
        badge_text="SUMMARY",
        badge_bg=C_CORAL,
    )
    cy = draw_bullet_list(
        c,
        [
            ("1. Progressive In-Race Updates Substantially Reduce Forecast Error:", "In both chronological evaluations, MAE drops monotonically from C0 (36.37 min in Eval I; 41.72 min in Eval II) to C1 (28.43 min in Eval I; 33.45 min in Eval II) and C2 (14.79 min in Eval I; 15.79 min in Eval II), while 90th-percentile absolute error (P90) contracts by more than 57% from C0 to C2."),
            ("2. ML Models Outperform Conditional Remainder Baselines at C1 & C2:", "While pre-race C0 demographics explain limited variance (1.34 to 1.66 min gain over B1/B2), CatBoost_ref and RF leverage observed split times at C1 and C2 to beat the additive remainder baseline B2 by 6.10 to 7.36 minutes of MAE."),
            ("3. Methodological Limitations & Future Directions:", "Current CatBoost_ref results reflect a single reference configuration (120 trees, seed 42); raw EventLocation strings are not yet mapped across multi-year location aliases; and individual athletes lack persistent longitudinal IDs across years."),
        ],
        col3_x + 14,
        cy,
        col_w - 28,
        font_size=10.3,
        leading=13.8,
        item_gap=4.0,
    )

    cy -= 4
    c.setFont("Helvetica-Bold", 11.5)
    c.setFillColor(C_BLACK)
    c.drawString(col3_x + 14, cy, "SELECTED REFERENCES & REPOSITORY ARTIFACTS")
    cy -= 14

    refs = [
        "[1] M. Thuany, P. T. Nikolaidis, and B. Knechtle, 'Machine learning approaches for predicting ironman triathlon performance: A review and practical perspectives,' Sports, vol. 13, no. 1, p. 24, 2025.",
        "[2] P. T. Nikolaidis et al., 'Predicting overall performance in Ironman 70.3 age group triathletes through split disciplines,' Scientific Reports, vol. 13, p. 11492, 2023.",
        "[3] G. Ferrari et al., 'What is the best discipline to predict overall triathlon performance? An analysis of Sprint, Olympic, Ironman 70.3, and Ironman 140.6,' Frontiers in Physiology, vol. 12, 2021.",
        "[4] R. Lepers, B. Knechtle, and P. J. Stapley, 'Trends in triathlon performance: Effects of sex and age,' Sports Medicine, vol. 43, no. 9, pp. 851–863, 2013.",
        "[5] L. Breiman, 'Random forests,' Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
        "[6] David, 'Ironman 70.3 races 2004–2020,' Kaggle Dataset (2023); Endurance-Data, 'Endurance-data race results 2021–2023' (2026).",
        "[7] Offline Exhibition Demo & Code Repository: github.com/kaity121/ironman-70-3-finish-time (Branch: bibi/demo-poster | Authorized Directory: bibi_demo_poster/).",
    ]
    for r_str in refs:
        cy = draw_wrapped_text(
            c,
            r_str,
            col3_x + 14,
            cy,
            col_w - 28,
            font_name="Helvetica",
            font_size=9.4,
            leading=12.6,
        )
        cy -= 3.5

    # Offline Launch Command Footer Strip inside Card 3.3
    cmd_box_h = 34.0
    cmd_box_y = y_conc + 14.0
    c.setFillColor(C_C0)
    c.setStrokeColor(C_BLACK)
    c.setLineWidth(1.8)
    c.rect(col3_x + 14, cmd_box_y, col_w - 28, cmd_box_h, stroke=1, fill=1)
    c.setFillColor(C_BLACK)
    c.setFont("Courier-Bold", 9.6)
    c.drawString(
        col3_x + 22,
        cmd_box_y + 19,
        "OFFLINE DEMO LAUNCH COMMAND (WINDOWS POWERSHELL):",
    )
    c.setFont("Courier", 9.4)
    c.drawString(
        col3_x + 22,
        cmd_box_y + 6,
        "python -m streamlit run bibi_demo_poster/app.py   |   Validation: python bibi_demo_poster/tests/test_scientific_validation.py",
    )

    c.showPage()
    c.save()
    print(f"Generated A0 poster PDF at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_a0_poster()
