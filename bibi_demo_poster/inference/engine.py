"""
Read-only Offline Inference Engine for IRONMAN 70.3 Finish Time Prediction.

Strictly loads original project assets in READ-ONLY mode from:
- config/run.json
- models/{split}_{checkpoint}.cbm
- models/{split}_{checkpoint}_baseline.csv
- metrics/example.csv
- metrics/all_models.csv
- metrics/cohort.csv
- metrics/before_consistency_check.csv
- metrics/split_consistency.csv
- metrics/snapshot_comparison.json

Never modifies, overwrites, or retrains any model or project file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Project root is two levels up from bibi_demo_poster/inference/engine.py
REPO_ROOT = Path(__file__).resolve().parents[2]

# Active country ISO2 codes verified from the trained CatBoost models + standard options
ACTIVE_COUNTRIES: List[Tuple[str, str]] = [
    ("US", "US — Hoa Kỳ"),
    ("VN", "VN — Việt Nam"),
    ("AU", "AU — Úc"),
    ("GB", "GB — Vương quốc Anh"),
    ("DE", "DE — Đức"),
    ("FR", "FR — Pháp"),
    ("CA", "CA — Canada"),
    ("CH", "CH — Thụy Sĩ"),
    ("AT", "AT — Áo"),
    ("BR", "BR — Brazil"),
    ("CN", "CN — Trung Quốc"),
    ("CO", "CO — Colombia"),
    ("DK", "DK — Đan Mạch"),
    ("EC", "EC — Ecuador"),
    ("ES", "ES — Tây Ban Nha"),
    ("ID", "ID — Indonesia"),
    ("IE", "IE — Ireland"),
    ("IN", "IN — Ấn Độ"),
    ("IT", "IT — Ý"),
    ("JP", "JP — Nhật Bản"),
    ("KR", "KR — Hàn Quốc"),
    ("MX", "MX — Mexico"),
    ("MY", "MY — Malaysia"),
    ("NA", "NA — Namibia (Bảo toàn mã ISO)"),
    ("NO", "NO — Na Uy"),
    ("NZ", "NZ — New Zealand"),
    ("PH", "PH — Philippines"),
    ("PL", "PL — Ba Lan"),
    ("PR", "PR — Puerto Rico"),
    ("SE", "SE — Thụy Điển"),
    ("SG", "SG — Singapore"),
    ("TH", "TH — Thái Lan"),
    ("TW", "TW — Đài Loan"),
    ("ZA", "ZA — Nam Phi"),
    ("BE", "BE — Bỉ"),
    ("__MISSING__", "__MISSING__ — Chưa xác định quốc gia"),
]

# Active IRONMAN 70.3 event locations verified directly against the CatBoost model trees
ACTIVE_EVENT_LOCATIONS: List[str] = [
    "IRONMAN 70.3 Florida",
    "IRONMAN 70.3 Vietnam",
    "IRONMAN 70.3 World Championship",
    "IRONMAN 70.3 Arizona",
    "IRONMAN 70.3 Atlantic City",
    "IRONMAN 70.3 Augusta",
    "IRONMAN 70.3 Austin",
    "IRONMAN 70.3 Barcelona",
    "IRONMAN 70.3 Bintan",
    "IRONMAN 70.3 Boise",
    "IRONMAN 70.3 Boulder",
    "IRONMAN 70.3 Brazil",
    "IRONMAN 70.3 Budapest",
    "IRONMAN 70.3 Buenos Aires",
    "IRONMAN 70.3 Buffalo Springs Lake",
    "IRONMAN 70.3 Busselton",
    "IRONMAN 70.3 Calgary",
    "IRONMAN 70.3 California",
    "IRONMAN 70.3 Cancun",
    "IRONMAN 70.3 Cartagena",
    "IRONMAN 70.3 Chattanooga",
    "IRONMAN 70.3 Cozumel",
    "IRONMAN 70.3 Dubai",
    "IRONMAN 70.3 Dublin",
    "IRONMAN 70.3 Eagleman",
    "IRONMAN 70.3 Gdynia",
    "IRONMAN 70.3 Gulf Coast",
    "IRONMAN 70.3 Hawaii",
    "IRONMAN 70.3 Hefei",
    "IRONMAN 70.3 Italy",
    "IRONMAN 70.3 Japan",
    "IRONMAN 70.3 Jonkoping",
    "IRONMAN 70.3 Kansas",
    "IRONMAN 70.3 Kraichgau",
    "IRONMAN 70.3 Kronborg",
    "IRONMAN 70.3 Lake Stevens",
    "IRONMAN 70.3 Lanzarote",
    "IRONMAN 70.3 Liuzhou",
    "IRONMAN 70.3 Los Cabos",
    "IRONMAN 70.3 Luxembourg",
    "IRONMAN 70.3 Maceio",
    "IRONMAN 70.3 Maine",
    "IRONMAN 70.3 Mallorca",
    "IRONMAN 70.3 Marbella",
    "IRONMAN 70.3 Miami",
    "IRONMAN 70.3 Mont-Tremblant",
    "IRONMAN 70.3 Monterrey",
    "IRONMAN 70.3 Muncie",
    "IRONMAN 70.3 Muskoka",
    "IRONMAN 70.3 New Orleans",
    "IRONMAN 70.3 Nice",
    "IRONMAN 70.3 Ohio",
    "IRONMAN 70.3 Panama",
    "IRONMAN 70.3 Philippines",
    "IRONMAN 70.3 Port Macquarie",
    "IRONMAN 70.3 Punta del Este",
    "IRONMAN 70.3 Putrajaya",
    "IRONMAN 70.3 Racine",
    "IRONMAN 70.3 Raleigh",
    "IRONMAN 70.3 Ruegen",
    "IRONMAN 70.3 San Juan",
    "IRONMAN 70.3 Santa Cruz",
    "IRONMAN 70.3 Santa Rosa",
    "IRONMAN 70.3 Singapore",
    "IRONMAN 70.3 South Africa",
    "IRONMAN 70.3 St. George",
    "IRONMAN 70.3 Staffordshire",
    "IRONMAN 70.3 Steelhead",
    "IRONMAN 70.3 Sunshine Coast",
    "IRONMAN 70.3 Superfrog",
    "IRONMAN 70.3 Switzerland",
    "IRONMAN 70.3 Syracuse",
    "IRONMAN 70.3 Taiwan",
    "IRONMAN 70.3 Texas",
    "IRONMAN 70.3 Thailand",
    "IRONMAN 70.3 Turkey",
    "IRONMAN 70.3 UK",
    "IRONMAN 70.3 Vichy",
    "IRONMAN 70.3 Victoria",
    "IRONMAN 70.3 Western Sydney",
    "IRONMAN 70.3 Xiamen",
    "IRONMAN 70.3 Zell am See-Kaprun",
]


# Exact split border ranges (in seconds) extracted directly from the 6 saved CatBoost .cbm models
MODEL_SPLIT_BORDER_BOUNDS: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {
    ("I", "C1"): {"swimtime": (1497.5, 4235.5)},
    ("I", "C2"): {"swimtime": (1582.5, 4235.5), "biketime": (8271.5, 15430.5)},
    ("II", "C1"): {"swimtime": (1582.5, 4172.5)},
    ("II", "C2"): {"swimtime": (1525.5, 4172.5), "biketime": (8096.5, 15427.5)},
}


def check_split_border_saturation(
    split: str,
    checkpoint: str,
    feature_name: str,
    value_seconds: Optional[float],
) -> Optional[Dict[str, Any]]:
    """
    Check if an input split time (swimtime or biketime) falls outside the quantized
    decision-tree split borders of the active CatBoost model ({split}_{checkpoint}.cbm).
    Returns None if inside the active split-border range, or a dict describing the
    boundary saturation when below min_border or above max_border.
    """
    if value_seconds is None or not np.isfinite(value_seconds) or value_seconds <= 0:
        return None
    bounds = MODEL_SPLIT_BORDER_BOUNDS.get((split, checkpoint), {}).get(feature_name)
    if bounds is None:
        return None
    min_b, max_b = bounds
    if value_seconds < min_b:
        return {
            "direction": "below_min",
            "min_border": min_b,
            "max_border": max_b,
            "value_seconds": float(value_seconds),
        }
    if value_seconds > max_b:
        return {
            "direction": "above_max",
            "min_border": min_b,
            "max_border": max_b,
            "value_seconds": float(value_seconds),
        }
    return None


def format_seconds_hms(seconds: Optional[float], signed: bool = False) -> str:
    """
    Format a duration in seconds as HH:MM:SS.
    Internally preserves exact seconds and rounds to nearest integer second for display.
    If signed=True, prepends '+' or '-' for non-zero differences.
    """
    if seconds is None or not np.isfinite(seconds):
        return "—"
    sign = ""
    if signed:
        if round(seconds) > 0:
            sign = "+"
        elif round(seconds) < 0:
            sign = "-"
    elif seconds < 0:
        sign = "-"

    total_sec = int(round(abs(float(seconds))))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}"


def hms_to_seconds(hours: int, minutes: int, seconds: int) -> float:
    """Convert hours, minutes, seconds integers into total seconds (float)."""
    return float(int(hours) * 3600 + int(minutes) * 60 + int(seconds))


def normalize_categorical(value: Any) -> str:
    """
    Reproduce exact categorical preprocessing from scripts/run.py (lines 40-41):
        data[col] = data[col].str.strip().replace('', '__MISSING__')
    Notice that 'NA' (Namibia) is preserved as 'NA', never converted to NaN.
    """
    if value is None:
        return "__MISSING__"
    s = str(value).strip()
    return "__MISSING__" if s == "" else s


def agegroup_to_ageband(agegroup: str) -> int:
    """
    Reproduce exact ageband derivation from scripts/run.py (line 39):
        data.agegroup.str.split('-').str[0].astype(int)
    """
    clean = str(agegroup).strip()
    return int(clean.split("-")[0])


def validate_athlete_inputs(
    checkpoint: str,
    gender: str,
    agegroup: str,
    countryiso: str,
    eventlocation: str,
    eventyear: int,
    valid_agegroups: List[str],
    swimtime_seconds: Optional[float] = None,
    biketime_seconds: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate user inputs for C0, C1, or C2 according to the project's cohort rules.
    Returns (is_valid, error_messages).
    """
    errors: List[str] = []

    if checkpoint not in {"C0", "C1", "C2"}:
        errors.append(f"Mốc kiểm tra '{checkpoint}' không hợp lệ. Phải là C0, C1 hoặc C2.")

    norm_gender = normalize_categorical(gender)
    if norm_gender not in {"M", "F"}:
        errors.append("Giới tính (gender) phải là 'M' (Nam) hoặc 'F' (Nữ).")

    if agegroup not in valid_agegroups:
        errors.append(f"Nhóm tuổi (agegroup) phải thuộc danh sách {valid_agegroups}.")

    try:
        yr = int(eventyear)
        if yr < 2004 or yr > 2030:
            errors.append("Năm thi đấu (eventyear) phải nằm trong khoảng từ 2004 đến 2030.")
    except (ValueError, TypeError):
        errors.append("Năm thi đấu (eventyear) phải là một số nguyên hợp lệ.")

    if checkpoint in {"C1", "C2"}:
        if swimtime_seconds is None or not np.isfinite(swimtime_seconds) or swimtime_seconds <= 0:
            errors.append("Thời gian bơi (swimtime) là bắt buộc tại mốc C1/C2 và phải lớn hơn 0 giây.")
        elif swimtime_seconds < 600 or swimtime_seconds > 10800:
            errors.append(
                f"Thời gian bơi ({format_seconds_hms(swimtime_seconds)}) nằm ngoài ngưỡng thông thường của IRONMAN 70.3 "
                "(00:10:00 đến 03:00:00). Vui lòng kiểm tra lại giá trị nhập."
            )

    if checkpoint == "C2":
        if biketime_seconds is None or not np.isfinite(biketime_seconds) or biketime_seconds <= 0:
            errors.append("Thời gian đạp xe (biketime) là bắt buộc tại mốc C2 và phải lớn hơn 0 giây.")
        elif biketime_seconds < 2400 or biketime_seconds > 25200:
            errors.append(
                f"Thời gian đạp xe ({format_seconds_hms(biketime_seconds)}) nằm ngoài ngưỡng thông thường của IRONMAN 70.3 "
                "(00:40:00 đến 07:00:00). Vui lòng kiểm tra lại giá trị nhập."
            )

    return (len(errors) == 0), errors


class IronmanInferenceEngine:
    """
    Read-only engine that loads config/run.json, all 6 CatBoost models (.cbm),
    all 6 baseline lookup tables (*_baseline.csv), and saved evaluation metrics.
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.config_path = self.root / "config" / "run.json"
        self.models_dir = self.root / "models"
        self.metrics_dir = self.root / "metrics"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing configuration file: {self.config_path}")

        self.config: Dict[str, Any] = json.loads(
            self.config_path.read_text(encoding="utf-8")
        )
        self.features_pre: List[str] = list(self.config["features_pre"])
        self.checkpoints: Dict[str, List[str]] = dict(self.config["checkpoints"])
        self.agegroups: List[str] = list(self.config["agegroups"])
        self.baseline_min_group: int = int(self.config["baseline_min_group"])
        self.evaluations: Dict[str, Any] = dict(self.config["evaluations"])

        self._models: Dict[Tuple[str, str], CatBoostRegressor] = {}
        self._baselines: Dict[Tuple[str, str], pd.DataFrame] = {}

        self._load_all_artifacts()

    def _load_all_artifacts(self) -> None:
        """Load all 6 CatBoost .cbm models and 6 baseline CSV files in read-only mode."""
        for split in ["I", "II"]:
            for cp in ["C0", "C1", "C2"]:
                model_file = self.models_dir / f"{split}_{cp}.cbm"
                baseline_file = self.models_dir / f"{split}_{cp}_baseline.csv"

                if not model_file.exists():
                    raise FileNotFoundError(f"Missing model file: {model_file}")
                if not baseline_file.exists():
                    raise FileNotFoundError(f"Missing baseline file: {baseline_file}")

                cb = CatBoostRegressor()
                cb.load_model(str(model_file))
                self._models[(split, cp)] = cb

                bdf = pd.read_csv(
                    baseline_file, keep_default_na=False, encoding="utf-8-sig"
                )
                bdf["size"] = pd.to_numeric(bdf["size"], errors="raise")
                bdf["median"] = pd.to_numeric(bdf["median"], errors="raise")
                self._baselines[(split, cp)] = bdf

    def get_feature_list(self, checkpoint: str) -> List[str]:
        """Return the exact ordered feature list for a given checkpoint (C0, C1, or C2)."""
        if checkpoint not in self.checkpoints:
            raise ValueError(f"Unknown checkpoint: {checkpoint}")
        return self.features_pre + list(self.checkpoints[checkpoint])

    def get_model(self, split: str, checkpoint: str) -> CatBoostRegressor:
        """Return the loaded CatBoostRegressor for (split, checkpoint)."""
        key = (split, checkpoint)
        if key not in self._models:
            raise KeyError(f"Model not found for split={split}, checkpoint={checkpoint}")
        return self._models[key]

    def get_baseline_table(self, split: str, checkpoint: str) -> pd.DataFrame:
        """Return a copy of the loaded baseline lookup table for (split, checkpoint)."""
        key = (split, checkpoint)
        if key not in self._baselines:
            raise KeyError(f"Baseline table not found for split={split}, checkpoint={checkpoint}")
        return self._baselines[key].copy()

    def prepare_feature_dataframe(
        self,
        checkpoint: str,
        gender: str,
        agegroup: str,
        countryiso: str,
        eventlocation: str,
        eventyear: int,
        swimtime_seconds: Optional[float] = None,
        biketime_seconds: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Prepare a single-row DataFrame matching scripts/run.py preprocessing and column ordering.
        """
        features = self.get_feature_list(checkpoint)
        norm_gender = normalize_categorical(gender)
        norm_country = normalize_categorical(countryiso)
        norm_location = normalize_categorical(eventlocation)
        ageband = agegroup_to_ageband(agegroup)

        row: Dict[str, Any] = {
            "gender": norm_gender,
            "ageband": int(ageband),
            "countryiso": norm_country,
            "eventlocation": norm_location,
            "eventyear": int(eventyear),
        }

        if checkpoint in {"C1", "C2"}:
            if swimtime_seconds is None or not np.isfinite(swimtime_seconds) or swimtime_seconds <= 0:
                raise ValueError(f"Valid positive swimtime_seconds required for {checkpoint}")
            row["swimtime"] = float(swimtime_seconds)

        if checkpoint == "C2":
            if biketime_seconds is None or not np.isfinite(biketime_seconds) or biketime_seconds <= 0:
                raise ValueError("Valid positive biketime_seconds required for C2")
            row["biketime"] = float(biketime_seconds)

        return pd.DataFrame([row], columns=features)

    def predict_baselines(
        self,
        split: str,
        checkpoint: str,
        gender: str,
        agegroup: str,
        swimtime_seconds: Optional[float] = None,
        biketime_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate B0, B1 (at C0), and B2 using the exact hierarchical fallback logic
        from scripts/run.py (lines 59-75):
            1. Start with 'all' level median remainder.
            2. Override with 'gender' level median remainder if size >= baseline_min_group (30).
            3. Override with 'gender+agegroup' level median remainder if size >= baseline_min_group (30).
            4. Add known split times (0 at C0, swimtime at C1, swimtime + biketime at C2).
            5. B0 is always the global 'all' median finish time from {split}_C0_baseline.csv.
        """
        norm_gender = normalize_categorical(gender)
        norm_agegroup = str(agegroup).strip()

        # B0 is always the fit cohort's global median finish time (stored in C0 baseline 'all' row)
        c0_table = self._baselines[(split, "C0")]
        b0_row = c0_table.loc[c0_table["level"] == "all"].iloc[0]
        b0_seconds = float(b0_row["median"])

        table = self._baselines[(split, checkpoint)]
        all_row = table.loc[table["level"] == "all"].iloc[0]
        base_remainder = float(all_row["median"])
        matched_level = "all"
        matched_group_size = int(all_row["size"])

        # Step 1: gender level
        g_rows = table.loc[
            (table["level"] == "gender")
            & (table["gender"] == norm_gender)
            & (table["size"] >= self.baseline_min_group)
        ]
        if not g_rows.empty:
            base_remainder = float(g_rows.iloc[0]["median"])
            matched_level = "gender"
            matched_group_size = int(g_rows.iloc[0]["size"])

        # Step 2: gender+agegroup level
        ga_rows = table.loc[
            (table["level"] == "gender+agegroup")
            & (table["gender"] == norm_gender)
            & (table["agegroup"] == norm_agegroup)
            & (table["size"] >= self.baseline_min_group)
        ]
        if not ga_rows.empty:
            base_remainder = float(ga_rows.iloc[0]["median"])
            matched_level = "gender+agegroup"
            matched_group_size = int(ga_rows.iloc[0]["size"])

        if checkpoint == "C0":
            known_seconds = 0.0
        elif checkpoint == "C1":
            if swimtime_seconds is None or swimtime_seconds <= 0:
                raise ValueError("Positive swimtime_seconds required for C1 baseline B2.")
            known_seconds = float(swimtime_seconds)
        elif checkpoint == "C2":
            if (
                swimtime_seconds is None
                or swimtime_seconds <= 0
                or biketime_seconds is None
                or biketime_seconds <= 0
            ):
                raise ValueError("Positive swimtime_seconds and biketime_seconds required for C2 baseline B2.")
            known_seconds = float(swimtime_seconds) + float(biketime_seconds)
        else:
            raise ValueError(f"Unknown checkpoint: {checkpoint}")

        b2_seconds = known_seconds + base_remainder

        result: Dict[str, Any] = {
            "B0": b0_seconds,
            "B2": b2_seconds,
            "known_split_seconds": known_seconds,
            "median_remainder_seconds": base_remainder,
            "matched_level": matched_level,
            "matched_group_size": matched_group_size,
        }
        if checkpoint == "C0":
            # In scripts/run.py line 75: if cp=='C0': preds['B1']=preds['B2']
            result["B1"] = b2_seconds

        return result

    def predict_checkpoint(
        self,
        split: str,
        checkpoint: str,
        gender: str,
        agegroup: str,
        countryiso: str,
        eventlocation: str,
        eventyear: int,
        swimtime_seconds: Optional[float] = None,
        biketime_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run live offline inference for a single checkpoint (C0, C1, or C2)
        using the saved CatBoost model ({split}_{checkpoint}.cbm) and corresponding baselines.
        """
        df_features = self.prepare_feature_dataframe(
            checkpoint=checkpoint,
            gender=gender,
            agegroup=agegroup,
            countryiso=countryiso,
            eventlocation=eventlocation,
            eventyear=eventyear,
            swimtime_seconds=swimtime_seconds,
            biketime_seconds=biketime_seconds,
        )
        model = self.get_model(split, checkpoint)
        catboost_pred_seconds = float(model.predict(df_features)[0])

        baselines = self.predict_baselines(
            split=split,
            checkpoint=checkpoint,
            gender=gender,
            agegroup=agegroup,
            swimtime_seconds=swimtime_seconds,
            biketime_seconds=biketime_seconds,
        )

        known_seconds = baselines["known_split_seconds"]
        implied_remaining_seconds = catboost_pred_seconds - known_seconds

        return {
            "split": split,
            "checkpoint": checkpoint,
            "model_file": f"models/{split}_{checkpoint}.cbm",
            "baseline_file": f"models/{split}_{checkpoint}_baseline.csv",
            "features_used": self.get_feature_list(checkpoint),
            "feature_values": df_features.iloc[0].to_dict(),
            "CatBoost_ref": catboost_pred_seconds,
            "B0": baselines["B0"],
            "B1": baselines.get("B1"),
            "B2": baselines["B2"],
            "known_split_seconds": known_seconds,
            "baseline_remainder_seconds": baselines["median_remainder_seconds"],
            "implied_remaining_seconds": implied_remaining_seconds,
            "matched_level": baselines["matched_level"],
            "matched_group_size": baselines["matched_group_size"],
        }

    def load_example_predictions(self) -> pd.DataFrame:
        """
        Load original saved test predictions from metrics/example.csv without modification.
        """
        path = self.metrics_dir / "example.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing saved examples file: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")

    def load_all_models_metrics(self) -> pd.DataFrame:
        """
        Load official evaluation metrics from metrics/all_models.csv.
        """
        path = self.metrics_dir / "all_models.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")

    def load_cohort_metrics(self) -> pd.DataFrame:
        """
        Load annual cohort statistics from metrics/cohort.csv.
        """
        path = self.metrics_dir / "cohort.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing cohort file: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")

    def load_before_qc_metrics(self) -> pd.DataFrame:
        """
        Load pre-QC sensitivity metrics from metrics/before_consistency_check.csv.
        """
        path = self.metrics_dir / "before_consistency_check.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing pre-QC file: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")

    def load_split_consistency(self) -> pd.DataFrame:
        """
        Load split consistency counts from metrics/split_consistency.csv.
        """
        path = self.metrics_dir / "split_consistency.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split consistency file: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")

    def load_snapshot_comparison(self) -> Dict[str, Any]:
        """
        Load Supabase vs. Drive CSV snapshot comparison from metrics/snapshot_comparison.json.
        """
        path = self.metrics_dir / "snapshot_comparison.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing snapshot comparison file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
