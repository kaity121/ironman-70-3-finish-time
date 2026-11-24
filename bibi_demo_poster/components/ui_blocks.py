"""
Neubrutalism + Sports Analytics Editorial UI Components for IRONMAN 70.3 Demo (Vietnamese Localization).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from inference import format_seconds_hms

THIS_DIR = Path(__file__).resolve().parent
ASSETS_DIR = THIS_DIR.parent / "assets"
LOGO_SVG_PATH = ASSETS_DIR / "ironman_official_logo.svg"

PALETTE = {
    "cream": "#FFFDF5",
    "black": "#000000",
    "white": "#FFFFFF",
    "C0": "#FFD23F",
    "C1": "#74B9FF",
    "C2": "#88D498",
    "accent": "#FF6B6B",
}

CHECKPOINT_LABELS = {
    "C0": ("C0 — TRƯỚC CUỘC ĐUA", "Chỉ sử dụng thông tin nhân khẩu học và thông tin giải đấu trước xuất phát"),
    "C1": ("C1 — SAU CHẶNG BƠI", "Thông tin trước cuộc đua + thời gian hoàn thành 1.9 km bơi (swimtime)"),
    "C2": ("C2 — SAU CHẶNG ĐẠP XE", "Thông tin trước cuộc đua + swimtime + thời gian hoàn thành 90 km đạp xe (biketime)"),
}


def _load_official_logo_svg() -> str:
    """Load the official IRONMAN Group M-Dot SVG from bibi_demo_poster/assets/."""
    if LOGO_SVG_PATH.exists():
        svg_text = LOGO_SVG_PATH.read_text(encoding="utf-8").strip()
        return " ".join(line.strip() for line in svg_text.splitlines() if line.strip())
    return ""


def render_header_banner(active_mode: str, active_split: str, active_cp: str) -> None:
    """Render top branding with official IRONMAN logo, main title, short explanation, and offline status indicator."""
    split_desc = (
        "Evaluation II (Huấn luyện 2004–2018 · Kiểm thử 2021–2023)"
        if active_split == "II"
        else "Evaluation I (Huấn luyện 2004–2016 · Kiểm thử 2019–2020)"
    )
    cp_badge_class = f"nb-badge-{active_cp.lower()}"
    cp_name_map = {
        "C0": "C0 — Trước cuộc đua",
        "C1": "C1 — Sau chặng bơi",
        "C2": "C2 — Sau chặng đạp xe",
    }
    logo_svg = _load_official_logo_svg()

    html = f"""
    <div class="nb-header-box">
        <div class="nb-header-top-row">
            <div class="nb-logo-brand-wrap">
                <div class="nb-official-logo-box" title="Official IRONMAN Group Logo (ironman.com/about/ironman-group)">
                    {logo_svg}
                </div>
                <div>
                    <span class="nb-badge nb-badge-black">PHÂN TÍCH THỂ THAO IRONMAN 70.3</span>
                    <span class="nb-badge nb-badge-c2">● CHẾ ĐỘ NGOẠI TUYẾN 100% (CHỈ ĐỌC)</span>
                </div>
            </div>
            <div>
                <span class="nb-badge">{split_desc}</span>
                <span class="nb-badge {cp_badge_class}">MỐC HIỆN TẠI: {cp_name_map.get(active_cp, active_cp)}</span>
            </div>
        </div>
        <div class="nb-title-main">
            DỰ ĐOÁN THỜI GIAN VỀ ĐÍCH THEO TIẾN TRÌNH CUỘC ĐUA IRONMAN 70.3
        </div>
        <p class="nb-subtitle">
            Hệ thống dự báo không rò rỉ dữ liệu thời gian tại 3 mốc: <b>C0 (Trước cuộc đua)</b>,
            <b>C1 (Sau chặng bơi 1.9 km)</b> và <b>C2 (Sau chặng đạp xe 90 km)</b> sử dụng các mô hình tham chiếu CatBoost đã lưu
            (<code>models/*.cbm</code>) kết hợp cùng các đường cơ sở trung vị phần thời gian còn lại (<code>Baseline B0, B1, B2</code>).
        </p>
    </div>
    """
    cleaned_html = "\n".join(line.lstrip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned_html, unsafe_allow_html=True)


def render_checkpoint_stepper(active_cp: str) -> None:
    """Render visual progression strip: TRƯỚC CUỘC ĐUA -> SAU CHẶNG BƠI -> SAU CHẶNG ĐẠP XE."""
    c0_cls = "nb-step-box nb-step-active-C0" if active_cp == "C0" else "nb-step-box"
    c1_cls = "nb-step-box nb-step-active-C1" if active_cp == "C1" else "nb-step-box"
    c2_cls = "nb-step-box nb-step-active-C2" if active_cp == "C2" else "nb-step-box"

    c0_state = "● MỐC ĐANG CHỌN" if active_cp == "C0" else "✓ ĐÃ CÓ DỮ LIỆU"
    c1_state = (
        "● MỐC ĐANG CHỌN"
        if active_cp == "C1"
        else ("✓ ĐÃ HOÀN THÀNH" if active_cp == "C2" else "○ BƯỚC 2")
    )
    c2_state = "● MỐC ĐANG CHỌN" if active_cp == "C2" else "○ BƯỚC 3"

    html = f"""
    <div class="nb-stepper-grid">
        <div class="{c0_cls}">
            <div class="nb-step-code">{c0_state} · MỐC C0</div>
            <div class="nb-step-title">TRƯỚC CUỘC ĐUA</div>
            <div class="nb-step-desc">5 đặc trưng: gender, ageband, countryiso, eventlocation, eventyear</div>
        </div>
        <div class="{c1_cls}">
            <div class="nb-step-code">{c1_state} · MỐC C1</div>
            <div class="nb-step-title">→ SAU CHẶNG BƠI (1.9 KM)</div>
            <div class="nb-step-desc">6 đặc trưng: Đặc trưng C0 + thời gian bơi thực tế swimtime (giây)</div>
        </div>
        <div class="{c2_cls}">
            <div class="nb-step-code">{c2_state} · MỐC C2</div>
            <div class="nb-step-title">→ SAU CHẶNG ĐẠP XE (90 KM)</div>
            <div class="nb-step-desc">7 đặc trưng: Đặc trưng C1 + thời gian đạp xe thực tế biketime (giây)</div>
        </div>
    </div>
    """
    cleaned_html = "\n".join(line.lstrip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned_html, unsafe_allow_html=True)


def render_hero_prediction_card(
    pred_result: Dict[str, Any],
    c0_pred_seconds: Optional[float] = None,
    c1_pred_seconds: Optional[float] = None,
) -> None:
    """
    Render the main FinishTime prediction card (7px 7px 0px #000000 hard shadow).
    """
    cp = pred_result["checkpoint"]
    split = pred_result["split"]
    cb_sec = float(pred_result["CatBoost_ref"])
    b2_sec = float(pred_result["B2"])
    b0_sec = float(pred_result["B0"])

    cp_title, _ = CHECKPOINT_LABELS[cp]

    deltas_html = ""
    if cp == "C1" and c0_pred_seconds is not None:
        diff_c0 = cb_sec - c0_pred_seconds
        deltas_html += (
            f'<span class="nb-badge">CHÊNH LỆCH SO VỚI C0: {format_seconds_hms(diff_c0, signed=True)} '
            f'({diff_c0/60:+.2f} phút)</span>'
        )
    elif cp == "C2":
        if c0_pred_seconds is not None:
            diff_c0 = cb_sec - c0_pred_seconds
            deltas_html += (
                f'<span class="nb-badge">CHÊNH LỆCH SO VỚI C0: {format_seconds_hms(diff_c0, signed=True)} '
                f'({diff_c0/60:+.2f} phút)</span>'
            )
        if c1_pred_seconds is not None:
            diff_c1 = cb_sec - c1_pred_seconds
            deltas_html += (
                f'<span class="nb-badge">CHÊNH LỆCH SO VỚI C1: {format_seconds_hms(diff_c1, signed=True)} '
                f'({diff_c1/60:+.2f} phút)</span>'
            )

    diff_b2 = cb_sec - b2_sec
    deltas_html += (
        f'<span class="nb-badge">SO VỚI BASELINE B2: {format_seconds_hms(diff_b2, signed=True)} '
        f'({diff_b2/60:+.2f} phút)</span>'
    )
    diff_b0 = cb_sec - b0_sec
    deltas_html += (
        f'<span class="nb-badge">SO VỚI BASELINE B0: {format_seconds_hms(diff_b0, signed=True)}</span>'
    )

    html = f"""
    <div class="nb-hero-card nb-hero-{cp}">
        <div class="nb-pred-label">
            THỜI GIAN VỀ ĐÍCH DỰ ĐOÁN · {cp_title} · MÔ HÌNH: CatBoost_ref ({pred_result["model_file"]}) · EVALUATION {split}
        </div>
        <div class="nb-pred-time">{format_seconds_hms(cb_sec)}</div>
        <div style="margin-bottom: 0.85rem;">
            <span class="nb-pred-seconds">GIÁ TRỊ DỰ ĐOÁN: {cb_sec:,.2f} GIÂY ({cb_sec/60:,.2f} PHÚT)</span>
        </div>
        <div>{deltas_html}</div>
    </div>
    """
    cleaned_html = "\n".join(line.lstrip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned_html, unsafe_allow_html=True)


def render_race_progress_card(
    checkpoint: str,
    pred_finish_seconds: float,
    swimtime_seconds: Optional[float] = None,
    biketime_seconds: Optional[float] = None,
) -> None:
    """
    Render completed vs. remaining race stages (Swim 1.9 km -> Bike 90 km -> Run 21.1 km).
    """
    if checkpoint == "C0":
        swim_status = "CHẶNG 1 · CHƯA BẮT ĐẦU"
        swim_cls = "nb-stage-item nb-stage-pending"
        swim_val = "Chưa xuất phát (Trước cuộc đua)"

        bike_status = "CHẶNG 2 · CHƯA BẮT ĐẦU"
        bike_cls = "nb-stage-item nb-stage-pending"
        bike_val = "Chưa xuất phát (Trước cuộc đua)"

        run_status = "CHẶNG 3 · CHƯA BẮT ĐẦU"
        run_cls = "nb-stage-item nb-stage-pending"
        run_val = f"Còn lại toàn bộ cự ly 70.3 ({format_seconds_hms(pred_finish_seconds)})"
    elif checkpoint == "C1":
        s_sec = float(swimtime_seconds or 0.0)
        rem_sec = pred_finish_seconds - s_sec
        swim_status = "✓ CHẶNG 1 · ĐÃ HOÀN THÀNH"
        swim_cls = "nb-stage-item nb-stage-done-swim"
        swim_val = f"{format_seconds_hms(s_sec)} ({s_sec:,.0f} giây)"

        bike_status = "CHẶNG 2 · CÒN LẠI"
        bike_cls = "nb-stage-item nb-stage-pending"
        bike_val = "Đang chuẩn bị / Đang thi đấu (Đạp xe 90 km)"

        run_status = "CHẶNG 2 + 3 CÒN LẠI (ĐẠP XE + CHẠY BỘ + CHUYỂN TIẾP)"
        run_cls = "nb-stage-item nb-stage-pending"
        run_val = f"Thời gian còn lại dự kiến: {format_seconds_hms(rem_sec)} ({rem_sec/60:,.1f} phút)"
    else:
        s_sec = float(swimtime_seconds or 0.0)
        b_sec = float(biketime_seconds or 0.0)
        rem_sec = pred_finish_seconds - (s_sec + b_sec)
        swim_status = "✓ CHẶNG 1 · ĐÃ HOÀN THÀNH"
        swim_cls = "nb-stage-item nb-stage-done-swim"
        swim_val = f"{format_seconds_hms(s_sec)} ({s_sec:,.0f} giây)"

        bike_status = "✓ CHẶNG 2 · ĐÃ HOÀN THÀNH"
        bike_cls = "nb-stage-item nb-stage-done-bike"
        bike_val = f"{format_seconds_hms(b_sec)} ({b_sec:,.0f} giây)"

        run_status = "→ CHẶNG CÒN LẠI: CHẠY BỘ 21.1 KM (+ CHUYỂN TIẾP)"
        run_cls = "nb-stage-item"
        run_val = (
            f"Thời gian còn lại dự kiến: {format_seconds_hms(rem_sec)} ({rem_sec/60:,.1f} phút)"
            if rem_sec > 0
            else f"Cảnh báo: Tổng chặng đã qua vượt dự đoán về đích ({format_seconds_hms(rem_sec, signed=True)})"
        )

    html = f"""
    <div class="nb-card">
        <div class="nb-section-header">
            <span>TIẾN TRÌNH CUỘC ĐUA (CHẶNG ĐÃ QUA & CHẶNG CÒN LẠI)</span>
            <span class="nb-badge">MỐC {checkpoint}</span>
        </div>
        <div class="nb-stage-grid">
            <div class="{swim_cls}">
                <div class="nb-step-code">{swim_status}</div>
                <div class="nb-step-title">BƠI 1.9 KM</div>
                <div class="nb-step-desc"><b>{swim_val}</b></div>
            </div>
            <div class="{bike_cls}">
                <div class="nb-step-code">{bike_status}</div>
                <div class="nb-step-title">ĐẠP XE 90 KM</div>
                <div class="nb-step-desc"><b>{bike_val}</b></div>
            </div>
            <div class="{run_cls}">
                <div class="nb-step-code">{run_status}</div>
                <div class="nb-step-title">CHẠY BỘ BÁN MARATHON 21.1 KM</div>
                <div class="nb-step-desc"><b>{run_val}</b></div>
            </div>
        </div>
    </div>
    """
    cleaned_html = "\n".join(line.lstrip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned_html, unsafe_allow_html=True)


def make_live_comparison_figure(
    checkpointResults: Dict[str, Dict[str, Any]],
    active_cp: str,
) -> plt.Figure:
    """
    Create a clean Neubrutalist Matplotlib horizontal bar chart comparing live predictions
    (B0, B1/B2, CatBoost_ref) across available checkpoints (Vietnamese labels).
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=140)
    fig.patch.set_facecolor(PALETTE["white"])
    ax.set_facecolor(PALETTE["cream"])

    res = checkpointResults[active_cp]
    labels: List[str] = []
    values_min: List[float] = []
    colors: List[str] = []
    hms_labels: List[str] = []

    labels.append("Baseline B0 (Trung vị toàn tập)")
    values_min.append(res["B0"] / 60.0)
    colors.append("#E2E0D8")
    hms_labels.append(format_seconds_hms(res["B0"]))

    if active_cp == "C0" and res.get("B1") is not None:
        labels.append("Baseline B1 (Trung vị Giới tính × Nhóm tuổi)")
        values_min.append(res["B1"] / 60.0)
        colors.append("#D0CEB8")
        hms_labels.append(format_seconds_hms(res["B1"]))

    labels.append(f"Baseline B2 ({active_cp}: Chặng đã biết + Trung vị còn lại)")
    values_min.append(res["B2"] / 60.0)
    colors.append("#B8B5A0")
    hms_labels.append(format_seconds_hms(res["B2"]))

    for prior_cp in ["C0", "C1", "C2"]:
        if prior_cp in checkpointResults:
            val_sec = checkpointResults[prior_cp]["CatBoost_ref"]
            is_curr = prior_cp == active_cp
            lbl = (
                f"★ CatBoost_ref @ {prior_cp} (MỐC ĐANG CHỌN)"
                if is_curr
                else f"CatBoost_ref @ {prior_cp} (Mốc trước)"
            )
            labels.append(lbl)
            values_min.append(val_sec / 60.0)
            colors.append(PALETTE[prior_cp])
            hms_labels.append(format_seconds_hms(val_sec))

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos,
        values_min,
        color=colors,
        edgecolor=PALETTE["black"],
        linewidth=2.0,
        height=0.62,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.2, fontweight="bold", color=PALETTE["black"])
    ax.invert_yaxis()
    ax.set_xlabel(
        "Thời gian về đích dự đoán (Phút & HH:MM:SS)",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["black"],
    )
    ax.set_title(
        f"SO SÁNH MÔ HÌNH SUY LUẬN TRỰC TIẾP TẠI MỐC {active_cp} (EVALUATION {res['split']})",
        fontsize=10.2,
        fontweight="bold",
        color=PALETTE["black"],
        pad=10,
    )

    max_v = max(values_min) if values_min else 360.0
    min_v = min(values_min) if values_min else 200.0
    ax.set_xlim(max(0, min_v - 45), max_v + 60)

    for bar, hms, vmin in zip(bars, hms_labels, values_min):
        ax.text(
            bar.get_width() + 2.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{hms} ({vmin:.1f} phút)",
            va="center",
            ha="left",
            fontsize=8.6,
            fontweight="bold",
            color=PALETTE["black"],
        )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(PALETTE["black"])
        ax.spines[spine].set_linewidth(2.0)

    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#000000")
    fig.tight_layout()
    return fig


def make_replay_chart(
    athlete_df: pd.DataFrame,
    active_cp: str,
    reveal_actual: bool,
) -> plt.Figure:
    """
    Create a Neubrutalist comparison chart for Real Test Replay (metrics/example.csv) in Vietnamese.
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=140)
    fig.patch.set_facecolor(PALETTE["white"])
    ax.set_facecolor(PALETTE["cream"])

    sub = athlete_df.loc[athlete_df["checkpoint"] == active_cp].copy()
    model_order = ["B0", "B1", "B2", "LR", "RF", "CatBoost_ref"]
    sub["order"] = sub["model"].apply(lambda m: model_order.index(m) if m in model_order else 99)
    sub = sub.sort_values("order")

    labels = []
    vals_min = []
    colors = []
    hms_list = []

    for _, row in sub.iterrows():
        m = str(row["model"])
        p_sec = float(row["pred_finish_seconds"])
        labels.append(f"{m} (seed={int(row['seed'])})")
        vals_min.append(p_sec / 60.0)
        hms_list.append(format_seconds_hms(p_sec))
        if m == "CatBoost_ref":
            colors.append(PALETTE[active_cp])
        elif m in {"LR", "RF"}:
            colors.append("#FFFFFF")
        else:
            colors.append("#E5E2D6")

    actual_sec = float(sub["actual_finish_seconds"].iloc[0])
    if reveal_actual:
        labels.append("★ THỜI GIAN VỀ ĐÍCH THỰC TẾ")
        vals_min.append(actual_sec / 60.0)
        hms_list.append(format_seconds_hms(actual_sec))
        colors.append(PALETTE["accent"])

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos,
        vals_min,
        color=colors,
        edgecolor=PALETTE["black"],
        linewidth=2.0,
        height=0.62,
    )

    if reveal_actual:
        ax.axvline(
            actual_sec / 60.0,
            color=PALETTE["accent"],
            linestyle="--",
            linewidth=2.2,
            label=f"Thực tế ({format_seconds_hms(actual_sec)})",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.2, fontweight="bold", color=PALETTE["black"])
    ax.invert_yaxis()
    ax.set_xlabel(
        "Thời gian về đích (Phút & HH:MM:SS)",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["black"],
    )
    rid = int(sub["resultid"].iloc[0])
    ax.set_title(
        f"DỰ ĐOÁN ĐÃ LƯU TRÊN TẬP KIỂM THỬ · RESULTID #{rid} · MỐC {active_cp} (EVALUATION II)",
        fontsize=10.2,
        fontweight="bold",
        color=PALETTE["black"],
        pad=10,
    )

    max_v = max(vals_min) if vals_min else 360.0
    min_v = min(vals_min) if vals_min else 200.0
    ax.set_xlim(max(0, min_v - 45), max_v + 65)

    for bar, hms, vmin in zip(bars, hms_list, vals_min):
        ax.text(
            bar.get_width() + 2.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{hms} ({vmin:.1f} phút)",
            va="center",
            ha="left",
            fontsize=8.6,
            fontweight="bold",
            color=PALETTE["black"],
        )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(PALETTE["black"])
        ax.spines[spine].set_linewidth(2.0)

    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#000000")
    fig.tight_layout()
    return fig


def make_global_mae_chart(all_models_df: pd.DataFrame, split: str) -> plt.Figure:
    """
    Plot official MAE (minutes) across C0, C1, C2 from metrics/all_models.csv (Vietnamese labels).
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=140)
    fig.patch.set_facecolor(PALETTE["white"])
    ax.set_facecolor(PALETTE["cream"])

    sub = all_models_df.loc[all_models_df["split"] == split].copy()
    checkpoints = ["C0", "C1", "C2"]
    models_to_plot = [
        ("B0", "Baseline B0 (Trung vị toàn tập)", "#CCCCCC"),
        ("B2", "Baseline B2 (Chặng + Trung vị nhóm)", "#8E8D8A"),
        ("LR", "Hồi quy tuyến tính (LR)", "#FFFFFF"),
        ("RF", "Rừng ngẫu nhiên (RF)", "#74B9FF"),
        ("CatBoost_ref", "CatBoost_ref (.cbm)", "#FFD23F"),
    ]

    x = np.arange(len(checkpoints))
    width = 0.15

    for idx, (m_code, m_label, color) in enumerate(models_to_plot):
        m_sub = sub.loc[sub["model"] == m_code].set_index("checkpoint")
        y_vals = [float(m_sub.loc[cp, "mae_minutes"]) for cp in checkpoints]
        offset = (idx - 2) * width
        bars = ax.bar(
            x + offset,
            y_vals,
            width=width,
            label=m_label,
            color=color,
            edgecolor=PALETTE["black"],
            linewidth=1.6,
        )
        for bar, val in zip(bars, y_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.6,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color=PALETTE["black"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["C0 — TRƯỚC CUỘC ĐUA", "C1 — SAU CHẶNG BƠI", "C2 — SAU CHẶNG ĐẠP XE"],
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["black"],
    )
    ax.set_ylabel("Sai số tuyệt đối trung bình — MAE (Phút)", fontsize=9.2, fontweight="bold")
    n_fit = int(sub["n_fit"].iloc[0])
    n_test = int(sub["n_test"].iloc[0])
    n_races = int(sub["races"].iloc[0])
    ax.set_title(
        f"SO SÁNH MAE (PHÚT) THEO MỐC DỰ ĐOÁN · EVALUATION {split} "
        f"(n_fit={n_fit:,} · n_test={n_test:,} · số giải={n_races})",
        fontsize=9.6,
        fontweight="bold",
        pad=10,
    )
    ax.set_ylim(0, max(sub["mae_minutes"]) + 8)
    ax.legend(
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=PALETTE["black"],
        fontsize=7.8,
        loc="upper right",
    )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(PALETTE["black"])
        ax.spines[spine].set_linewidth(2.0)

    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#000000")
    fig.tight_layout()
    return fig
