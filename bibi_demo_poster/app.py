"""
IRONMAN 70.3 Finish Time Prediction — Ứng dụng Mô phỏng Ngoại tuyến (Giao diện Tiếng Việt).

Hệ thống thiết kế: NEUBRUTALISM + SPORTS ANALYTICS EDITORIAL
Kiến trúc: Chạy ngoại tuyến 100% (Truy cập chỉ đọc vào dữ liệu và mô hình gốc của dự án)

Hai chế độ hoạt động:
1. Chế độ A — Mô phỏng trực tiếp (C0 -> C1 -> C2 theo tiến trình cuộc đua)
2. Chế độ B — Xem lại dữ liệu kiểm thử (10 vận động viên kiểm thử thực tế từ metrics/example.csv)

Lệnh khởi chạy:
    streamlit run bibi_demo_poster/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import streamlit as st

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from components import (
    make_global_mae_chart,
    make_live_comparison_figure,
    make_replay_chart,
    render_checkpoint_stepper,
    render_header_banner,
    render_hero_prediction_card,
    render_race_progress_card,
)
from inference import (
    ACTIVE_COUNTRIES,
    ACTIVE_EVENT_LOCATIONS,
    IronmanInferenceEngine,
    check_split_border_saturation,
    format_seconds_hms,
    hms_to_seconds,
    validate_athlete_inputs,
)
from styles import NEUBRUTAL_CSS

st.set_page_config(
    page_title="IRONMAN 70.3 · Dự đoán thời gian về đích",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Force module cache invalidation across code updates on Streamlit Cloud
import importlib
import styles.neubrutalism as _nb_styles_mod
import components.ui_blocks as _nb_ui_mod
import inference.engine as _nb_eng_mod

importlib.reload(_nb_styles_mod)
importlib.reload(_nb_eng_mod)
importlib.reload(_nb_ui_mod)

from components.ui_blocks import (
    make_global_mae_chart,
    make_live_comparison_figure,
    make_replay_chart,
    render_checkpoint_stepper,
    render_header_banner,
    render_hero_prediction_card,
    render_race_progress_card,
)
from inference.engine import (
    ACTIVE_COUNTRIES,
    ACTIVE_EVENT_LOCATIONS,
    IronmanInferenceEngine,
    check_split_border_saturation,
    format_seconds_hms,
    hms_to_seconds,
    validate_athlete_inputs,
)
from styles.neubrutalism import NEUBRUTAL_CSS

# Apply Neubrutalist Editorial CSS
st.markdown(NEUBRUTAL_CSS, unsafe_allow_html=True)


def render_html(html_str: str) -> None:
    """
    Strip leading indentation from each line so Markdown never misinterprets
    nested HTML strings as 4-space indented `<pre><code>` blocks.
    """
    cleaned = "\n".join(line.lstrip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine() -> IronmanInferenceEngine:
    """Load the read-only inference engine once per session."""
    return IronmanInferenceEngine(REPO_ROOT)


engine = get_engine()


def init_session_state() -> None:
    """Initialize session state with clean defaults."""
    defaults: Dict[str, Any] = {
        "demo_mode": "LIVE_SIMULATION",  # "LIVE_SIMULATION" or "REAL_TEST_REPLAY"
        "eval_split": "II",  # "II" (Fit 2004-2018, Test 2021-2023) or "I" (Fit 2004-2016, Test 2019-2020)
        "live_cp": "C0",  # "C0", "C1", or "C2"
        "gender": "F",
        "agegroup": "35-39",
        "countryiso": "US",
        "eventlocation": "IRONMAN 70.3 Florida",
        "custom_location": "",
        "eventyear": 2022,
        "swim_h": 0,
        "swim_m": 35,
        "swim_s": 0,
        "bike_h": 2,
        "bike_m": 40,
        "bike_s": 0,
        "computed_results": {},  # Persisted predictions per checkpoint: {"C0": res, "C1": res, "C2": res}
        "computed_signatures": {},  # Persisted input signatures per checkpoint
        "replay_cp": "C0",
        "replay_resultid": 840060,
        "replay_reveal_actual": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_demo_state() -> None:
    """Reset all interactive states to default exhibition values."""
    keys_to_clear = [
        "demo_mode",
        "eval_split",
        "live_cp",
        "gender",
        "agegroup",
        "countryiso",
        "eventlocation",
        "custom_location",
        "eventyear",
        "swim_h",
        "swim_m",
        "swim_s",
        "bike_h",
        "bike_m",
        "bike_s",
        "computed_results",
        "computed_signatures",
        "replay_cp",
        "replay_resultid",
        "replay_reveal_actual",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    init_session_state()


init_session_state()

active_cp_for_header = (
    st.session_state["live_cp"]
    if st.session_state["demo_mode"] == "LIVE_SIMULATION"
    else st.session_state["replay_cp"]
)

# 1–3. Header, Official IRONMAN Logo, Title, Explanation, and Offline Mode Indicator
render_header_banner(
    active_mode=st.session_state["demo_mode"],
    active_split=st.session_state["eval_split"],
    active_cp=active_cp_for_header,
)

# 4. Top Navigation Bar: Mode Switcher + Evaluation Cohort Selector + Reset Demo
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1.3, 1.35, 1.65, 0.85])

with nav_col1:
    mode_a_label = (
        "● CHẾ ĐỘ A: MÔ PHỎNG TRỰC TIẾP"
        if st.session_state["demo_mode"] == "LIVE_SIMULATION"
        else "○ CHẾ ĐỘ A: MÔ PHỎNG TRỰC TIẾP"
    )
    if st.button(mode_a_label, key="btn_mode_a", use_container_width=True):
        st.session_state["demo_mode"] = "LIVE_SIMULATION"
        st.rerun()

with nav_col2:
    mode_b_label = (
        "● CHẾ ĐỘ B: XEM LẠI DỮ LIỆU KIỂM THỬ"
        if st.session_state["demo_mode"] == "REAL_TEST_REPLAY"
        else "○ CHẾ ĐỘ B: XEM LẠI DỮ LIỆU KIỂM THỬ"
    )
    if st.button(mode_b_label, key="btn_mode_b", use_container_width=True):
        st.session_state["demo_mode"] = "REAL_TEST_REPLAY"
        st.rerun()

with nav_col3:
    split_options = {
        "II": "Evaluation II · Huấn luyện 2004–18 / Kiểm thử 2021–23 (II_*.cbm)",
        "I": "Evaluation I · Huấn luyện 2004–16 / Kiểm thử 2019–20 (I_*.cbm)",
    }
    selected_split = st.selectbox(
        "NHÓM ĐÁNH GIÁ & MÔ HÌNH ĐÃ LƯU",
        options=["II", "I"],
        index=0 if st.session_state["eval_split"] == "II" else 1,
        format_func=lambda s: split_options[s],
        disabled=(st.session_state["demo_mode"] == "REAL_TEST_REPLAY"),
        help="Chọn giai đoạn đánh giá theo thời gian để nạp đúng mô hình CatBoost (I_C*.cbm hoặc II_C*.cbm) và bảng Baseline. Chế độ Xem lại dữ liệu kiểm thử sử dụng Evaluation II theo đúng dữ liệu trong metrics/example.csv.",
    )
    if (
        st.session_state["demo_mode"] == "LIVE_SIMULATION"
        and selected_split != st.session_state["eval_split"]
    ):
        st.session_state["eval_split"] = selected_split
        st.session_state["eventyear"] = 2022 if selected_split == "II" else 2019
        st.rerun()

with nav_col4:
    st.markdown("<div style='height: 1.68rem;'></div>", unsafe_allow_html=True)
    if st.button("↺ ĐẶT LẠI MÔ PHỎNG", key="btn_reset_demo", type="primary", use_container_width=True):
        reset_demo_state()
        st.rerun()


# =============================================================================
# CHẾ ĐỘ A: MÔ PHỎNG TRỰC TIẾP (C0 -> C1 -> C2)
# =============================================================================
if st.session_state["demo_mode"] == "LIVE_SIMULATION":
    current_cp = st.session_state["live_cp"]
    current_split = st.session_state["eval_split"]

    # 5. C0 / C1 / C2 Checkpoint Navigation
    render_checkpoint_stepper(current_cp)

    cp_btn1, cp_btn2, cp_btn3 = st.columns(3)
    with cp_btn1:
        b_c0 = "● CHỌN C0 — TRƯỚC CUỘC ĐUA" if current_cp == "C0" else "CHỌN C0 — TRƯỚC CUỘC ĐUA"
        if st.button(b_c0, key="btn_live_c0", use_container_width=True):
            st.session_state["live_cp"] = "C0"
            st.rerun()
    with cp_btn2:
        b_c1 = "● CHỌN C1 — SAU CHẶNG BƠI" if current_cp == "C1" else "CHỌN C1 — SAU CHẶNG BƠI"
        if st.button(b_c1, key="btn_live_c1", use_container_width=True):
            st.session_state["live_cp"] = "C1"
            st.rerun()
    with cp_btn3:
        b_c2 = "● CHỌN C2 — SAU CHẶNG ĐẠP XE" if current_cp == "C2" else "CHỌN C2 — SAU CHẶNG ĐẠP XE"
        if st.button(b_c2, key="btn_live_c2", use_container_width=True):
            st.session_state["live_cp"] = "C2"
            st.rerun()

    # 6 & 7. Left Column: Athlete Information & Input Section | Right Column: Main Prediction Card & Race Progress
    left_col, right_col = st.columns([1.05, 1.35], gap="large")

    with left_col:
        render_html(
            f"""
            <div class="nb-card">
                <div class="nb-section-header">
                    <span>THÔNG TIN VẬN ĐỘNG VIÊN & ĐẦU VÀO ({current_cp})</span>
                    <span class="nb-badge nb-badge-{current_cp.lower()}">MÔ HÌNH: {current_split}_{current_cp}.cbm</span>
                </div>
                <p style="font-size: 0.88rem; margin-bottom: 0.5rem;">
                    Chỉ các đặc trưng thực sự được mô hình <code>models/{current_split}_{current_cp}.cbm</code> yêu cầu tại mốc <b>{current_cp}</b> mới được mở bên dưới.
                    Thời gian về đích thực tế không bao giờ được yêu cầu hoặc dùng làm đầu vào.
                </p>
            </div>
            """
        )

        # Quick Athlete Presets for Exhibition Visitors
        pr_c1, pr_c2, pr_c3 = st.columns(3)
        with pr_c1:
            if st.button("MẪU 1: F 35-39 US", key="pre_1", use_container_width=True):
                st.session_state["gender"] = "F"
                st.session_state["agegroup"] = "35-39"
                st.session_state["countryiso"] = "US"
                st.session_state["eventlocation"] = "IRONMAN 70.3 Florida"
                st.session_state["eventyear"] = 2022 if current_split == "II" else 2019
                st.session_state["swim_h"], st.session_state["swim_m"], st.session_state["swim_s"] = 0, 35, 0
                st.session_state["bike_h"], st.session_state["bike_m"], st.session_state["bike_s"] = 2, 42, 0
                st.rerun()
        with pr_c2:
            if st.button("MẪU 2: M 30-34 VN", key="pre_2", use_container_width=True):
                st.session_state["gender"] = "M"
                st.session_state["agegroup"] = "30-34"
                st.session_state["countryiso"] = "VN"
                st.session_state["eventlocation"] = "IRONMAN 70.3 Vietnam"
                st.session_state["eventyear"] = 2023 if current_split == "II" else 2019
                st.session_state["swim_h"], st.session_state["swim_m"], st.session_state["swim_s"] = 0, 32, 30
                st.session_state["bike_h"], st.session_state["bike_m"], st.session_state["bike_s"] = 2, 28, 0
                st.rerun()
        with pr_c3:
            if st.button("MẪU 3: M 50-54 AT", key="pre_3", use_container_width=True):
                st.session_state["gender"] = "M"
                st.session_state["agegroup"] = "50-54"
                st.session_state["countryiso"] = "AT"
                st.session_state["eventlocation"] = "IRONMAN 70.3 Zell am See-Kaprun"
                st.session_state["eventyear"] = 2022 if current_split == "II" else 2019
                st.session_state["swim_h"], st.session_state["swim_m"], st.session_state["swim_s"] = 0, 38, 15
                st.session_state["bike_h"], st.session_state["bike_m"], st.session_state["bike_s"] = 2, 51, 30
                st.rerun()

        # Pre-race C0 Features (Always Required at C0, C1, C2)
        in_c1, in_c2 = st.columns(2)
        with in_c1:
            gender_val = st.selectbox(
                "1. GIỚI TÍNH (gender)",
                options=["F", "M"],
                index=0 if st.session_state["gender"] == "F" else 1,
                format_func=lambda g: "F — Nữ (Age-Group)" if g == "F" else "M — Nam (Age-Group)",
            )
            st.session_state["gender"] = gender_val

        with in_c2:
            ag_list = engine.agegroups
            ag_idx = ag_list.index(st.session_state["agegroup"]) if st.session_state["agegroup"] in ag_list else 3
            agegroup_val = st.selectbox(
                "2. NHÓM TUỔI → CẬN DƯỚI (ageband)",
                options=ag_list,
                index=ag_idx,
                format_func=lambda ag: f"{ag} (ageband = {ag.split('-')[0]})",
            )
            st.session_state["agegroup"] = agegroup_val

        country_codes = [c[0] for c in ACTIVE_COUNTRIES]
        country_labels = {c[0]: c[1] for c in ACTIVE_COUNTRIES}
        c_idx = country_codes.index(st.session_state["countryiso"]) if st.session_state["countryiso"] in country_codes else 0
        country_val = st.selectbox(
            "3. MÃ QUỐC GIA VẬN ĐỘNG VIÊN (countryiso)",
            options=country_codes,
            index=c_idx,
            format_func=lambda code: country_labels.get(code, code),
        )
        st.session_state["countryiso"] = country_val

        custom_loc_option = "[Tùy chỉnh / Nhập địa điểm giải đấu khác...]"
        loc_options = ACTIVE_EVENT_LOCATIONS + [custom_loc_option]
        loc_idx = (
            loc_options.index(st.session_state["eventlocation"])
            if st.session_state["eventlocation"] in loc_options
            else 0
        )
        selected_loc = st.selectbox(
            "4. ĐỊA ĐIỂM GIẢI ĐẤU (eventlocation)",
            options=loc_options,
            index=loc_idx,
        )
        if selected_loc == custom_loc_option:
            custom_loc = st.text_input(
                "NHẬP TÊN ĐỊA ĐIỂM GIẢI ĐẤU (Tự động cắt khoảng trắng; để trống → __MISSING__)",
                value=st.session_state["custom_location"],
                placeholder="Ví dụ: IRONMAN 70.3 Tasmania",
            )
            st.session_state["custom_location"] = custom_loc
            final_location = custom_loc
        else:
            st.session_state["eventlocation"] = selected_loc
            final_location = selected_loc

        year_val = st.number_input(
            "5. NĂM THI ĐẤU (eventyear)",
            min_value=2004,
            max_value=2030,
            value=int(st.session_state["eventyear"]),
            step=1,
            help="Evaluation I huấn luyện: 2004–2016 (kiểm thử 2019–2020). Evaluation II huấn luyện: 2004–2018 (kiểm thử 2021–2023).",
        )
        st.session_state["eventyear"] = int(year_val)

        # Conditional Stage Split Inputs: C1 (SwimTime) and C2 (SwimTime + BikeTime)
        swim_seconds_val = None
        bike_seconds_val = None

        if current_cp in {"C1", "C2"}:
            render_html(
                """
                <div class="nb-card" style="background-color: #74B9FF; padding: 0.75rem 1rem; margin-top: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.86rem;">
                        MỞ KHÓA MỐC C1 (SAU CHẶNG BƠI) · NHẬP THỜI GIAN BƠI 1.9 KM THỰC TẾ (HH:MM:SS)
                    </span>
                </div>
                """
            )
            sw_c1, sw_c2, sw_c3 = st.columns(3)
            with sw_c1:
                sh = st.number_input("GIỜ BƠI", min_value=0, max_value=3, value=int(st.session_state["swim_h"]), step=1)
                st.session_state["swim_h"] = int(sh)
            with sw_c2:
                sm = st.number_input("PHÚT BƠI", min_value=0, max_value=59, value=int(st.session_state["swim_m"]), step=1)
                st.session_state["swim_m"] = int(sm)
            with sw_c3:
                ss = st.number_input("GIÂY BƠI", min_value=0, max_value=59, value=int(st.session_state["swim_s"]), step=1)
                st.session_state["swim_s"] = int(ss)

            swim_seconds_val = hms_to_seconds(sh, sm, ss)
            st.caption(
                f"Thời gian bơi đã nhập (`swimtime`): **{format_seconds_hms(swim_seconds_val)}** = **`{swim_seconds_val:,.0f}` giây** "
                f"({swim_seconds_val/60:,.2f} phút)"
            )
            swim_sat = check_split_border_saturation(
                current_split, current_cp, "swimtime", swim_seconds_val
            )
            if swim_sat is not None and 600 <= swim_seconds_val <= 10800:
                min_s, max_s = swim_sat["min_border"], swim_sat["max_border"]
                bound_desc = (
                    f"vượt ngưỡng chia tối đa (<code>{max_s:,.1f} giây</code> / <code>{format_seconds_hms(max_s)}</code>)"
                    if swim_sat["direction"] == "above_max"
                    else f"nhỏ hơn ngưỡng chia tối thiểu (<code>{min_s:,.1f} giây</code> / <code>{format_seconds_hms(min_s)}</code>)"
                )
                render_html(
                    f"""
                    <div class="nb-saturation-note">
                        <b>⚠ CHÚ THÍCH KỸ THUẬT (BÃO HÒA NGƯỠNG CHIA MÔ HÌNH):</b> Giá trị <code>swimtime = {swim_seconds_val:,.0f} giây</code>
                        hợp lệ để chạy mô phỏng nhưng {bound_desc} của cây quyết định
                        <code>{current_split}_{current_cp}.cbm</code> (phạm vi ngưỡng chia: <code>{min_s:,.1f}–{max_s:,.1f} giây</code>).
                        Do đó <code>CatBoost_ref</code> giữ nguyên giá trị dự đoán tại nút lá biên (không phải lỗi giới hạn thời gian thi đấu).
                    </div>
                    """
                )

        if current_cp == "C2":
            render_html(
                """
                <div class="nb-card" style="background-color: #88D498; padding: 0.75rem 1rem; margin-top: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.86rem;">
                        MỞ KHÓA MỐC C2 (SAU CHẶNG ĐẠP XE) · NHẬP THỜI GIAN ĐẠP XE 90 KM THỰC TẾ (HH:MM:SS)
                    </span>
                </div>
                """
            )
            bk_c1, bk_c2, bk_c3 = st.columns(3)
            with bk_c1:
                bh = st.number_input("GIỜ ĐẠP XE", min_value=0, max_value=7, value=int(st.session_state["bike_h"]), step=1)
                st.session_state["bike_h"] = int(bh)
            with bk_c2:
                bm = st.number_input("PHÚT ĐẠP XE", min_value=0, max_value=59, value=int(st.session_state["bike_m"]), step=1)
                st.session_state["bike_m"] = int(bm)
            with bk_c3:
                bs = st.number_input("GIÂY ĐẠP XE", min_value=0, max_value=59, value=int(st.session_state["bike_s"]), step=1)
                st.session_state["bike_s"] = int(bs)

            bike_seconds_val = hms_to_seconds(bh, bm, bs)
            st.caption(
                f"Thời gian đạp xe đã nhập (`biketime`): **{format_seconds_hms(bike_seconds_val)}** = **`{bike_seconds_val:,.0f}` giây** "
                f"({bike_seconds_val/60:,.2f} phút)"
            )
            bike_sat = check_split_border_saturation(
                current_split, current_cp, "biketime", bike_seconds_val
            )
            if bike_sat is not None and 2400 <= bike_seconds_val <= 25200:
                min_b, max_b = bike_sat["min_border"], bike_sat["max_border"]
                bound_desc_b = (
                    f"vượt ngưỡng chia tối đa (<code>{max_b:,.1f} giây</code> / <code>{format_seconds_hms(max_b)}</code>)"
                    if bike_sat["direction"] == "above_max"
                    else f"nhỏ hơn ngưỡng chia tối thiểu (<code>{min_b:,.1f} giây</code> / <code>{format_seconds_hms(min_b)}</code>)"
                )
                render_html(
                    f"""
                    <div class="nb-saturation-note">
                        <b>⚠ CHÚ THÍCH KỸ THUẬT (BÃO HÒA NGƯỠNG CHIA MÔ HÌNH):</b> Giá trị <code>biketime = {bike_seconds_val:,.0f} giây</code>
                        hợp lệ để chạy mô phỏng nhưng {bound_desc_b} của cây quyết định
                        <code>{current_split}_{current_cp}.cbm</code> (phạm vi ngưỡng chia: <code>{min_b:,.1f}–{max_b:,.1f} giây</code>).
                        Do đó <code>CatBoost_ref</code> giữ nguyên giá trị dự đoán tại nút lá biên (không phải lỗi giới hạn thời gian thi đấu).
                    </div>
                    """
                )

        # Explicit Action Button for Checkpoint Inference
        is_valid, validation_errors = validate_athlete_inputs(
            checkpoint=current_cp,
            gender=st.session_state["gender"],
            agegroup=st.session_state["agegroup"],
            countryiso=st.session_state["countryiso"],
            eventlocation=final_location,
            eventyear=st.session_state["eventyear"],
            valid_agegroups=engine.agegroups,
            swimtime_seconds=swim_seconds_val,
            biketime_seconds=bike_seconds_val,
        )

        def _build_cp_signature(cp_name: str) -> tuple:
            base = (
                current_split,
                cp_name,
                st.session_state["gender"],
                st.session_state["agegroup"],
                st.session_state["countryiso"],
                final_location.strip(),
                int(st.session_state["eventyear"]),
            )
            if cp_name == "C0":
                return base
            if cp_name == "C1":
                return base + (float(swim_seconds_val or 0.0),)
            return base + (float(swim_seconds_val or 0.0), float(bike_seconds_val or 0.0))

        current_sig = _build_cp_signature(current_cp)

        st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
        calc_btn_label = f"TÍNH DỰ ĐOÁN {current_cp}"
        calc_clicked = st.button(
            calc_btn_label,
            key=f"btn_calc_{current_cp}",
            type="primary",
            use_container_width=True,
        )
        if calc_clicked and is_valid:
            c0_computed = engine.predict_checkpoint(
                split=current_split,
                checkpoint="C0",
                gender=st.session_state["gender"],
                agegroup=st.session_state["agegroup"],
                countryiso=st.session_state["countryiso"],
                eventlocation=final_location,
                eventyear=st.session_state["eventyear"],
            )
            st.session_state["computed_results"]["C0"] = c0_computed
            st.session_state["computed_signatures"]["C0"] = _build_cp_signature("C0")

            if current_cp in {"C1", "C2"} and swim_seconds_val is not None:
                c1_computed = engine.predict_checkpoint(
                    split=current_split,
                    checkpoint="C1",
                    gender=st.session_state["gender"],
                    agegroup=st.session_state["agegroup"],
                    countryiso=st.session_state["countryiso"],
                    eventlocation=final_location,
                    eventyear=st.session_state["eventyear"],
                    swimtime_seconds=swim_seconds_val,
                )
                st.session_state["computed_results"]["C1"] = c1_computed
                st.session_state["computed_signatures"]["C1"] = _build_cp_signature("C1")

            if current_cp == "C2" and swim_seconds_val is not None and bike_seconds_val is not None:
                c2_computed = engine.predict_checkpoint(
                    split=current_split,
                    checkpoint="C2",
                    gender=st.session_state["gender"],
                    agegroup=st.session_state["agegroup"],
                    countryiso=st.session_state["countryiso"],
                    eventlocation=final_location,
                    eventyear=st.session_state["eventyear"],
                    swimtime_seconds=swim_seconds_val,
                    biketime_seconds=bike_seconds_val,
                )
                st.session_state["computed_results"]["C2"] = c2_computed
                st.session_state["computed_signatures"]["C2"] = _build_cp_signature("C2")

        # Quick Advance Checkpoint Button
        if current_cp == "C0":
            if st.button("→ HOÀN THÀNH BƠI & CHUYỂN SANG MỐC C1 (SAU CHẶNG BƠI)", key="adv_c1", use_container_width=True):
                st.session_state["live_cp"] = "C1"
                st.rerun()
        elif current_cp == "C1":
            if st.button("→ HOÀN THÀNH ĐẠP XE & CHUYỂN SANG MỐC C2 (SAU CHẶNG ĐẠP XE)", key="adv_c2", use_container_width=True):
                st.session_state["live_cp"] = "C2"
                st.rerun()

    # Right Column: Validation, Main Prediction Card, Race Progress, and Model Comparison
    with right_col:
        if not is_valid:
            err_items = "".join(f"<li>{e}</li>" for e in validation_errors)
            render_html(
                f"""
                <div class="nb-card" style="background-color: #FF6B6B;">
                    <div class="nb-section-header">LỖI KIỂM TRA DỮ LIỆU ĐẦU VÀO</div>
                    <ul style="margin: 0; padding-left: 1.25rem; font-weight: 700;">{err_items}</ul>
                </div>
                """
            )

        stored_results: Dict[str, Dict[str, Any]] = st.session_state["computed_results"]
        stored_sigs: Dict[str, tuple] = st.session_state["computed_signatures"]
        has_result_for_cp = current_cp in stored_results
        is_stale = has_result_for_cp and (stored_sigs.get(current_cp) != current_sig)

        if not has_result_for_cp:
            render_html(
                f"""
                <div class="nb-pending-box">
                    <div style="margin-bottom: 0.65rem;">
                        <span class="nb-badge nb-badge-{current_cp.lower()}">MỐC {current_cp} · CHỜ XÁC NHẬN TÍNH TOÁN</span>
                    </div>
                    <div style="font-family: var(--font-heading); font-size: 1.35rem; font-weight: 800; margin-bottom: 0.5rem;">
                        Nhập đầy đủ thông tin và nhấn nút để tính thời gian về đích dự đoán.
                    </div>
                    <p style="font-family: var(--font-mono); font-size: 0.86rem; margin: 0;">
                        Nhấn nút <b>"{calc_btn_label}"</b> ở cột bên trái để thực hiện suy luận với mô hình <code>models/{current_split}_{current_cp}.cbm</code> và bảng <code>Baseline</code>.
                    </p>
                </div>
                """
            )
        else:
            if is_stale:
                render_html(
                    f"""
                    <div class="nb-stale-banner">
                        ⚠ Dữ liệu đã thay đổi – Vui lòng tính lại dự đoán (Nhấn nút "{calc_btn_label}" để cập nhật kết quả mới).
                    </div>
                    """
                )

            active_res = stored_results[current_cp]
            c0_res_stored = stored_results.get("C0")
            c1_res_stored = stored_results.get("C1")

            # 7. Main FinishTime Prediction Card (Displays persisted result until recalculated)
            render_hero_prediction_card(
                pred_result=active_res,
                c0_pred_seconds=c0_res_stored["CatBoost_ref"] if (current_cp in {"C1", "C2"} and c0_res_stored is not None) else None,
                c1_pred_seconds=c1_res_stored["CatBoost_ref"] if (current_cp == "C2" and c1_res_stored is not None) else None,
            )

            # 8. Race Progress Visualization (Uses the persisted feature values of the displayed prediction)
            stored_feats = active_res["feature_values"]
            render_race_progress_card(
                checkpoint=current_cp,
                pred_finish_seconds=active_res["CatBoost_ref"],
                swimtime_seconds=stored_feats.get("swimtime"),
                biketime_seconds=stored_feats.get("biketime"),
            )

            # Exact Feature Vector Passed to CatBoost Model
            feat_badges = "".join(
                f'<span class="nb-badge">{k} = <b>{v}</b></span>'
                for k, v in active_res["feature_values"].items()
            )
            render_html(
                f"""
                <div class="nb-card">
                    <div class="nb-section-header">
                        <span>VÉC-TƠ ĐẶC TRƯNG ĐẦU VÀO TRUYỀN TỚI {active_res["model_file"]}</span>
                        <span class="nb-badge">{len(active_res["features_used"])} ĐẶC TRƯNG</span>
                    </div>
                    <div>{feat_badges}</div>
                </div>
                """
            )

    # 9. Model Comparison Section (Live Inference vs Baselines + Saved Historical Reference)
    if current_cp in st.session_state["computed_results"]:
        results_by_cp = {
            k: v
            for k, v in st.session_state["computed_results"].items()
            if (current_cp == "C0" and k == "C0")
            or (current_cp == "C1" and k in {"C0", "C1"})
            or (current_cp == "C2" and k in {"C0", "C1", "C2"})
        }
        active_res = results_by_cp[current_cp]
        rows_html = ""
        b0_s = active_res["B0"]
        rows_html += (
            f"<tr>"
            f"<td>Baseline B0 (Trung vị toàn tập huấn luyện)</td>"
            f"<td>{current_cp}</td>"
            f"<td>Tra cứu trực tiếp ({active_res['baseline_file']})</td>"
            f"<td>{format_seconds_hms(b0_s)}</td>"
            f"<td>{b0_s:,.1f} giây</td>"
            f"<td>{format_seconds_hms(active_res['CatBoost_ref'] - b0_s, signed=True)}</td>"
            f"</tr>"
        )
        if current_cp == "C0" and active_res.get("B1") is not None:
            b1_s = active_res["B1"]
            rows_html += (
                f"<tr>"
                f"<td>Baseline B1 (Trung vị Giới tính × Nhóm tuổi trước đua)</td>"
                f"<td>C0</td>"
                f"<td>Tra cứu nhóm {active_res['matched_level']} (n={active_res['matched_group_size']:,})</td>"
                f"<td>{format_seconds_hms(b1_s)}</td>"
                f"<td>{b1_s:,.1f} giây</td>"
                f"<td>{format_seconds_hms(active_res['CatBoost_ref'] - b1_s, signed=True)}</td>"
                f"</tr>"
            )
        b2_s = active_res["B2"]
        rows_html += (
            f"<tr>"
            f"<td>Baseline B2 (Tổng chặng đã biết + Trung vị phần còn lại)</td>"
            f"<td>{current_cp}</td>"
            f"<td>Tra cứu nhóm {active_res['matched_level']} (n={active_res['matched_group_size']:,})</td>"
            f"<td>{format_seconds_hms(b2_s)}</td>"
            f"<td>{b2_s:,.1f} giây</td>"
            f"<td>{format_seconds_hms(active_res['CatBoost_ref'] - b2_s, signed=True)}</td>"
            f"</tr>"
        )

        for cp_key in ["C0", "C1", "C2"]:
            if cp_key in results_by_cp:
                r_cp = results_by_cp[cp_key]
                cb_val = r_cp["CatBoost_ref"]
                row_cls = f"nb-row-highlight-{cp_key}" if cp_key == current_cp else ""
                delta_str = (
                    "MỐC ĐANG CHỌN"
                    if cp_key == current_cp
                    else format_seconds_hms(active_res["CatBoost_ref"] - cb_val, signed=True)
                )
                rows_html += (
                    f'<tr class="{row_cls}">'
                    f"<td>★ CatBoost_ref ({r_cp['model_file']})</td>"
                    f"<td>{cp_key}</td>"
                    f"<td>Suy luận trực tiếp CatBoost (seed=42)</td>"
                    f"<td>{format_seconds_hms(cb_val)}</td>"
                    f"<td>{cb_val:,.2f} giây</td>"
                    f"<td>{delta_str}</td>"
                    f"</tr>"
                )

        render_html(
            f"""
            <div class="nb-card">
                <div class="nb-section-header">
                    <span>SO SÁNH MÔ HÌNH (SUY LUẬN TRỰC TIẾP CATBOOST_REF VÀ BASELINE B0, B1, B2)</span>
                    <span class="nb-badge nb-badge-black">CHẾ ĐỘ NGOẠI TUYẾN</span>
                </div>
                <div class="nb-table-wrap">
                    <table class="nb-table">
                        <thead>
                            <tr>
                                <th style="width: 27%;">Phương pháp / Mô hình</th>
                                <th style="width: 7%;">Mốc</th>
                                <th style="width: 24%;">Nguồn suy luận</th>
                                <th style="width: 14%;">Thời gian về đích dự đoán</th>
                                <th style="width: 14%;">Số giây chính xác</th>
                                <th style="width: 14%;">Chênh lệch với CB hiện tại</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
                <p style="font-size: 0.82rem; font-family: var(--font-mono); margin: 0;">
                    <b>Ghi chú tính toàn vẹn khoa học:</b> Trong kho mã nguồn, chỉ có 6 mô hình CatBoost (<code>models/*.cbm</code>) và 6 bảng tra cứu Baseline
                    (<code>models/*_baseline.csv</code>) được lưu trữ để suy luận trực tiếp. Hai mô hình Hồi quy tuyến tính (<code>LR</code>) và Rừng ngẫu nhiên (<code>RF</code>)
                    không lưu file trọng số trong <code>models/</code>, do đó hệ thống tuyệt đối không tự tạo dự đoán LR/RF giả lập ở chế độ trực tiếp.
                    Để đối chiếu kết quả dự đoán LR và RF đã lưu trên các vận động viên thực tế, vui lòng chuyển sang <b>Chế độ B: Xem lại dữ liệu kiểm thử</b>.
                </p>
            </div>
            """
        )

        fig_live = make_live_comparison_figure(results_by_cp, current_cp)
        st.pyplot(fig_live, use_container_width=True)
        plt.close(fig_live)


# =============================================================================
# CHẾ ĐỘ B: XEM LẠI DỮ LIỆU KIỂM THỬ (10 LƯỢT THI TRONG metrics/example.csv)
# =============================================================================
else:
    ex_df = engine.load_example_predictions()
    replay_cp = st.session_state["replay_cp"]
    replay_rid = st.session_state["replay_resultid"]
    reveal_actual = bool(st.session_state["replay_reveal_actual"])

    # 5. Checkpoint Navigation for Replay Mode
    render_checkpoint_stepper(replay_cp)

    rcp_1, rcp_2, rcp_3 = st.columns(3)
    with rcp_1:
        lbl0 = "● XEM MỐC C0 — TRƯỚC CUỘC ĐUA" if replay_cp == "C0" else "XEM MỐC C0 — TRƯỚC CUỘC ĐUA"
        if st.button(lbl0, key="btn_rep_c0", use_container_width=True):
            st.session_state["replay_cp"] = "C0"
            st.rerun()
    with rcp_2:
        lbl1 = "● XEM MỐC C1 — SAU CHẶNG BƠI" if replay_cp == "C1" else "XEM MỐC C1 — SAU CHẶNG BƠI"
        if st.button(lbl1, key="btn_rep_c1", use_container_width=True):
            st.session_state["replay_cp"] = "C1"
            st.rerun()
    with rcp_3:
        lbl2 = "● XEM MỐC C2 — SAU CHẶNG ĐẠP XE" if replay_cp == "C2" else "XEM MỐC C2 — SAU CHẶNG ĐẠP XE"
        if st.button(lbl2, key="btn_rep_c2", use_container_width=True):
            st.session_state["replay_cp"] = "C2"
            st.rerun()

    rep_left, rep_right = st.columns([0.95, 1.45], gap="large")

    with rep_left:
        render_html(
            """
            <div class="nb-card">
                <div class="nb-section-header">
                    <span>THÔNG TIN VẬN ĐỘNG VIÊN KIỂM THỬ</span>
                    <span class="nb-badge nb-badge-black">metrics/example.csv</span>
                </div>
                <p style="font-size: 0.88rem; margin-bottom: 0.5rem;">
                    Trình chiếu 10 lượt thi đấu thực tế trong tập kiểm thử từ file <code>metrics/example.csv</code> (Evaluation II, năm 2021, mã giải RaceID 774).
                    Toàn bộ kết quả dự đoán (<code>B0, B1, B2, LR, RF, CatBoost_ref</code>) được đọc nguyên bản từ dữ liệu đã lưu, không tái tạo hay chỉnh sửa.
                </p>
            </div>
            """
        )

        available_rids = sorted(ex_df["resultid"].unique().tolist())
        rid_idx = available_rids.index(replay_rid) if replay_rid in available_rids else 0
        chosen_rid = st.selectbox(
            "CHỌN BẢN GHI VẬN ĐỘNG VIÊN KIỂM THỬ (resultid)",
            options=available_rids,
            index=rid_idx,
            format_func=lambda r: f"Vận động viên #{available_rids.index(r)+1:02d} · resultid={r} (Giải 774 · Năm 2021)",
        )
        if chosen_rid != replay_rid:
            st.session_state["replay_resultid"] = int(chosen_rid)
            st.session_state["replay_reveal_actual"] = False
            st.rerun()

        athlete_rows = ex_df.loc[ex_df["resultid"] == st.session_state["replay_resultid"]].copy()
        race_id_val = int(athlete_rows["raceid"].iloc[0])
        event_yr_val = int(athlete_rows["eventyear"].iloc[0])
        actual_sec_val = float(athlete_rows["actual_finish_seconds"].iloc[0])

        # Explicit Reveal Actual Finish Time Control
        render_html(
            """
            <div class="nb-card" style="margin-top: 0.75rem;">
                <div class="nb-section-header">
                    <span>ĐIỀU KHIỂN THỜI GIAN VỀ ĐÍCH THỰC TẾ</span>
                    <span class="nb-badge nb-badge-accent">MỞ Ở BƯỚC CUỐI</span>
                </div>
                <p style="font-size: 0.86rem; margin-bottom: 0.5rem;">
                    Theo đúng quy trình trình diễn nghiên cứu (<i>"Khi làm demo chỉ mở actual ở bước cuối"</i>),
                    thời gian về đích thực tế của vận động viên được ẩn mặc định cho đến khi bạn nhấn nút hiển thị bên dưới.
                </p>
            </div>
            """
        )

        if not reveal_actual:
            if st.button("HIỂN THỊ THỜI GIAN VỀ ĐÍCH THỰC TẾ", key="btn_reveal_actual", type="primary", use_container_width=True):
                st.session_state["replay_reveal_actual"] = True
                st.rerun()
        else:
            if st.button("ẨN THỜI GIAN VỀ ĐÍCH THỰC TẾ", key="btn_hide_actual", use_container_width=True):
                st.session_state["replay_reveal_actual"] = False
                st.rerun()

        if reveal_actual:
            render_html(
                f"""
                <div class="nb-hero-card" style="background-color: #FF6B6B; padding: 1.1rem 1.4rem; margin-top: 0.75rem;">
                    <div class="nb-pred-label">★ THỜI GIAN VỀ ĐÍCH THỰC TẾ · RESULTID #{chosen_rid}</div>
                    <div style="font-family: var(--font-mono); font-size: 2.6rem; font-weight: 700;">{format_seconds_hms(actual_sec_val)}</div>
                    <span class="nb-pred-seconds">THỰC TẾ CHÍNH XÁC: {actual_sec_val:,.1f} GIÂY ({actual_sec_val/60:,.2f} PHÚT)</span>
                </div>
                """
            )
        else:
            render_html(
                """
                <div class="nb-card" style="background-color: #FFFDF5; border-style: dashed; margin-top: 0.75rem;">
                    <span style="font-family: var(--font-mono); font-weight: 700;">🔒 THỜI GIAN VỀ ĐÍCH THỰC TẾ: ĐANG ẨN (Nhấn "HIỂN THỊ THỜI GIAN VỀ ĐÍCH THỰC TẾ" ở trên để mở)</span>
                </div>
                """
            )

    with rep_right:
        cp_rows = athlete_rows.loc[athlete_rows["checkpoint"] == replay_cp].copy()
        cb_row = cp_rows.loc[cp_rows["model"] == "CatBoost_ref"].iloc[0]
        cb_pred_sec = float(cb_row["pred_finish_seconds"])

        c0_cb_sec = float(
            athlete_rows.loc[
                (athlete_rows["checkpoint"] == "C0") & (athlete_rows["model"] == "CatBoost_ref"),
                "pred_finish_seconds",
            ].iloc[0]
        )
        c1_cb_sec = float(
            athlete_rows.loc[
                (athlete_rows["checkpoint"] == "C1") & (athlete_rows["model"] == "CatBoost_ref"),
                "pred_finish_seconds",
            ].iloc[0]
        )
        c2_cb_sec = float(
            athlete_rows.loc[
                (athlete_rows["checkpoint"] == "C2") & (athlete_rows["model"] == "CatBoost_ref"),
                "pred_finish_seconds",
            ].iloc[0]
        )

        err_badge = ""
        if reveal_actual:
            err_sec = cb_pred_sec - actual_sec_val
            err_badge = (
                f'<span class="nb-badge nb-badge-accent">'
                f'SAI SỐ SO VỚI THỰC TẾ: {format_seconds_hms(err_sec, signed=True)} ({err_sec/60:+.2f} phút | |MAE|={abs(err_sec)/60:.2f} phút)'
                f'</span>'
            )

        # Main Hero Card for Saved Replay Record
        render_html(
            f"""
            <div class="nb-hero-card nb-hero-{replay_cp}">
                <div class="nb-pred-label">
                    THỜI GIAN VỀ ĐÍCH DỰ ĐOÁN (ĐÃ LƯU) · RESULTID #{chosen_rid} · GIẢI #{race_id_val} ({event_yr_val}) · MỐC {replay_cp} · CatBoost_ref (seed=42)
                </div>
                <div class="nb-pred-time">{format_seconds_hms(cb_pred_sec)}</div>
                <div style="margin-bottom: 0.75rem;">
                    <span class="nb-pred-seconds">GIÁ TRỊ ĐÃ LƯU: {cb_pred_sec:,.8f} GIÂY ({cb_pred_sec/60:,.2f} PHÚT)</span>
                </div>
                <div>
                    <span class="nb-badge">C0 CatBoost_ref: {format_seconds_hms(c0_cb_sec)}</span>
                    <span class="nb-badge">C1 CatBoost_ref: {format_seconds_hms(c1_cb_sec)}</span>
                    <span class="nb-badge">C2 CatBoost_ref: {format_seconds_hms(c2_cb_sec)}</span>
                    {err_badge}
                </div>
            </div>
            """
        )

        fig_rep = make_replay_chart(athlete_rows, replay_cp, reveal_actual)
        st.pyplot(fig_rep, use_container_width=True)
        plt.close(fig_rep)

    # Full-width card for the 7-column SAVED PREDICTIONS AT CHECKPOINT table
    model_order = ["B0", "B1", "B2", "LR", "RF", "CatBoost_ref"]
    cp_rows["order"] = cp_rows["model"].apply(lambda m: model_order.index(m) if m in model_order else 99)
    cp_rows = cp_rows.sort_values("order")

    t_rows_html = ""
    for _, r in cp_rows.iterrows():
        m_name = str(r["model"])
        m_seed = int(r["seed"])
        p_sec = float(r["pred_finish_seconds"])
        row_cls = f"nb-row-highlight-{replay_cp}" if m_name == "CatBoost_ref" else ""

        if reveal_actual:
            diff_s = p_sec - actual_sec_val
            actual_col = format_seconds_hms(actual_sec_val)
            err_col = f"{format_seconds_hms(diff_s, signed=True)} ({abs(diff_s)/60:.2f} phút)"
        else:
            actual_col = "🔒 ĐANG ẨN"
            err_col = "🔒 ĐANG ẨN"

        t_rows_html += (
            f'<tr class="{row_cls}">'
            f"<td>{m_name}</td>"
            f"<td>{m_seed}</td>"
            f"<td>{replay_cp}</td>"
            f"<td>{format_seconds_hms(p_sec)}</td>"
            f"<td>{p_sec:,.2f} giây</td>"
            f"<td>{actual_col}</td>"
            f"<td>{err_col}</td>"
            f"</tr>"
        )

    render_html(
        f"""
        <div class="nb-card">
            <div class="nb-section-header">
                <span>SO SÁNH MÔ HÌNH ĐÃ LƯU TẠI MỐC {replay_cp} (RESULTID #{chosen_rid})</span>
                <span class="nb-badge">{len(cp_rows)} PHƯƠNG PHÁP ĐÃ LƯU</span>
            </div>
            <div class="nb-table-wrap">
                <table class="nb-table">
                    <thead>
                        <tr>
                            <th style="width: 18%;">Mô hình</th>
                            <th style="width: 8%;">Seed</th>
                            <th style="width: 8%;">Mốc</th>
                            <th style="width: 16%;">Thời gian dự đoán</th>
                            <th style="width: 16%;">Số giây</th>
                            <th style="width: 14%;">Thực tế</th>
                            <th style="width: 20%;">Sai số (|MAE|)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {t_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """
    )

    # Full-width card for C0 -> C1 -> C2 Progression Matrix
    pivot_rows_html = ""
    for m_name in ["B0", "B1", "B2", "LR", "RF", "CatBoost_ref"]:
        m_sub = athlete_rows.loc[athlete_rows["model"] == m_name].set_index("checkpoint")
        if m_sub.empty:
            continue
        c0_v = format_seconds_hms(float(m_sub.loc["C0", "pred_finish_seconds"])) if "C0" in m_sub.index else "Chỉ có ở C0"
        c1_v = format_seconds_hms(float(m_sub.loc["C1", "pred_finish_seconds"])) if "C1" in m_sub.index else "—"
        c2_v = format_seconds_hms(float(m_sub.loc["C2", "pred_finish_seconds"])) if "C2" in m_sub.index else "—"
        hl = f"nb-row-highlight-{replay_cp}" if m_name == "CatBoost_ref" else ""
        if reveal_actual and "C2" in m_sub.index:
            c2_err = abs(float(m_sub.loc["C2", "pred_finish_seconds"]) - actual_sec_val) / 60.0
            c2_err_str = f"{c2_err:.2f} phút"
        elif reveal_actual and "C0" in m_sub.index:
            c0_err = abs(float(m_sub.loc["C0", "pred_finish_seconds"]) - actual_sec_val) / 60.0
            c2_err_str = f"{c0_err:.2f} phút (C0)"
        else:
            c2_err_str = "🔒 ĐANG ẨN"

        pivot_rows_html += (
            f'<tr class="{hl}">'
            f"<td>{m_name}</td>"
            f"<td>{c0_v}</td>"
            f"<td>{c1_v}</td>"
            f"<td>{c2_v}</td>"
            f"<td>{c2_err_str}</td>"
            f"</tr>"
        )

    render_html(
        f"""
        <div class="nb-card">
            <div class="nb-section-header">
                <span>MA TRẬN TIẾN TRÌNH DỰ ĐOÁN C0 → C1 → C2 (RESULTID #{chosen_rid})</span>
                <span class="nb-badge">EVALUATION II</span>
            </div>
            <div class="nb-table-wrap">
                <table class="nb-table">
                    <thead>
                        <tr>
                            <th>Mô hình</th>
                            <th>C0 (Trước cuộc đua)</th>
                            <th>C1 (Sau chặng bơi)</th>
                            <th>C2 (Sau chặng đạp xe)</th>
                            <th>Sai số tuyệt đối cuối</th>
                        </tr>
                    </thead>
                    <tbody>
                        {pivot_rows_html}
                    </tbody>
                </table>
            </div>
            <p style="font-size: 0.80rem; font-family: var(--font-mono); margin: 0;">
                Ghi chú: Theo thiết kế thực nghiệm gốc (<code>scripts/run.py</code>), <code>Baseline B1</code> được định nghĩa riêng tại mốc trước cuộc đua
                <code>C0</code> (nơi <code>B1 == B2</code>). Tại mốc <code>C1</code> và <code>C2</code>, đường cơ sở trung vị phần còn lại theo nhân khẩu học được báo cáo dưới tên <code>B2</code>.
            </p>
        </div>
        """
    )


# =============================================================================
# 10. THÔNG TIN PHƯƠNG PHÁP NGHIÊN CỨU & KẾT QUẢ ĐÁNH GIÁ (metrics/all_models.csv)
# =============================================================================
st.markdown("<hr style='border: 2px solid #000000; margin: 1.5rem 0;'>", unsafe_allow_html=True)

with st.expander("📘 KẾT QUẢ THỰC NGHIỆM TOÀN TẬP & PHƯƠNG PHÁP NGHIÊN CỨU (metrics/all_models.csv)", expanded=False):
    all_models_df = engine.load_all_models_metrics()
    cohort_df = engine.load_cohort_metrics()
    snap_info = engine.load_snapshot_comparison()

    m_col1, m_col2 = st.columns(2, gap="large")
    with m_col1:
        fig_m2 = make_global_mae_chart(all_models_df, "II")
        st.pyplot(fig_m2, use_container_width=True)
        plt.close(fig_m2)
    with m_col2:
        fig_m1 = make_global_mae_chart(all_models_df, "I")
        st.pyplot(fig_m1, use_container_width=True)
        plt.close(fig_m1)

    disp_df = all_models_df.copy()
    disp_df["mae_minutes"] = disp_df["mae_minutes"].map(lambda x: f"{x:.2f} phút")
    disp_df["rmse_minutes"] = disp_df["rmse_minutes"].map(lambda x: f"{x:.2f} phút")
    disp_df["bias_minutes"] = disp_df["bias_minutes"].map(lambda x: f"{x:+.2f} phút")
    disp_df["p90_abs_minutes"] = disp_df["p90_abs_minutes"].map(lambda x: f"{x:.2f} phút")

    render_html(
        f"""
        <div class="nb-card">
            <div class="nb-section-header">
                <span>TÓM TẮT PHƯƠNG PHÁP NGHIÊN CỨU ĐÃ KIỂM CHỨNG</span>
                <span class="nb-badge nb-badge-c2">1,152,658 BẢN GHI HỢP LỆ</span>
            </div>
            <ul style="font-size: 0.9rem; line-height: 1.5; margin-bottom: 0;">
                <li><b>Bộ dữ liệu thống nhất & Đối chiếu Snapshot:</b> {snap_info['drive_eligible']:,} lượt hoàn thành hợp lệ thuộc nhóm tuổi Age-Group qua {int(cohort_df['races'].sum()):,} giải đấu (2004–2023), khớp tuyệt đối 100% giữa bộ CSV trên Google Drive và view <code>vw_valid_c1c2</code> trên Supabase.</li>
                <li><b>Hai giai đoạn đánh giá theo thời gian:</b> <b>Evaluation I</b> huấn luyện trên giai đoạn 2004–2016 (n = 495,489) và kiểm thử trên 2019–2020 (n = 125,912 qua 127 giải). <b>Evaluation II</b> huấn luyện trên 2004–2018 (n = 697,267) và kiểm thử trên 2021–2023 (n = 329,366 sau bước lọc QC qua 258 giải; loại bỏ 113 bản ghi có <code>FinishTime &lt;= SwimTime + BikeTime</code>).</li>
                <li><b>Các mô hình Baseline:</b> <code>B0</code> = trung vị FinishTime toàn tập huấn luyện; <code>B1</code> = trung vị trước cuộc đua theo <code>gender × agegroup</code>; <code>B2</code> = tổng thời gian các chặng đã hoàn thành + trung vị phần thời gian còn lại theo <code>gender × agegroup</code> (lùi về <code>gender</code> rồi <code>all</code> nếu cỡ mẫu nhóm &lt; 30).</li>
                <li><b>Cấu hình tham chiếu CatBoost (<code>CatBoost_ref</code>) & Chỉ số Đánh giá (<code>MAE, RMSE, P90</code>):</b> 120 vòng lặp (iterations), độ sâu cây (depth) = 6, tốc độ học (learning_rate) = 0.08, L2 leaf reg = 3, hàm mất mát RMSE, random_seed = 42. Bảng dưới báo cáo đầy đủ <code>MAE</code>, <code>RMSE</code>, độ lệch (bias) và phân vị 90 sai số tuyệt đối (<code>P90</code>) theo đơn vị phút.</li>
            </ul>
        </div>
        """
    )
    st.dataframe(disp_df, use_container_width=True, hide_index=True)
