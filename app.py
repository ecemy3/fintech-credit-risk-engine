from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Credit Risk Analysis Dashboard",
    page_icon="CR",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_FILES: List[str] = [
    "model_metrics_long.csv",
    "model_comparison.csv",
    "feature_importance_gbt.csv",
    "feature_importance_rf.csv",
    "confusion_matrix_best.csv",
    "roc_curve_best.csv",
    "threshold_tuning_best_model.csv",
    "business_cost_matrix.csv",
    "kpi_cards.csv",
    "loan_status_distribution.csv",
    "monthly_default_trend.csv",
    "fico_default_risk.csv",
    "home_ownership_risk.csv",
    "loan_purpose_risk.csv",
]


THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg": "#0B1220",
        "bg_alt": "#101A2E",
        "card_bg": "#13263A",
        "text_primary": "#EAF2FF",
        "text_secondary": "#9EB3C9",
        "accent_blue": "#2D8CFF",
        "risk_red": "#E85D5D",
        "success_green": "#2FBF71",
        "warning_orange": "#F2A93B",
        "grid": "#2A3B54",
    },
    "light": {
        "bg": "#EFF4FB",
        "bg_alt": "#DDE8F8",
        "card_bg": "#FFFFFF",
        "text_primary": "#0E1D33",
        "text_secondary": "#4F647A",
        "accent_blue": "#1D6FD8",
        "risk_red": "#C83E3E",
        "success_green": "#1F9D5C",
        "warning_orange": "#D88A1D",
        "grid": "#D7E1EF",
    },
}


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(255, 255, 255, {alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def apply_theme_css(theme: Dict[str, str]) -> None:
    card_bg_glass = hex_to_rgba(theme["card_bg"], 0.74)
    border_glass = hex_to_rgba(theme["accent_blue"], 0.35)
    sidebar_glass = hex_to_rgba(theme["card_bg"], 0.85)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at top right, {theme['bg_alt']} 0%, {theme['bg']} 65%);
            color: {theme['text_primary']};
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {sidebar_glass} 0%, {theme['bg']} 100%);
            border-right: 1px solid {hex_to_rgba(theme['accent_blue'], 0.25)};
        }}

        div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] p {{
            color: {theme['text_primary']};
        }}

        .main-title {{
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.3px;
            color: {theme['text_primary']};
            margin-bottom: 0.2rem;
        }}

        .main-subtitle {{
            color: {theme['text_secondary']};
            font-size: 0.98rem;
            margin-bottom: 1.2rem;
        }}

        .kpi-card {{
            background: {card_bg_glass};
            border: 1px solid {border_glass};
            border-radius: 16px;
            padding: 1rem 0.95rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 26px {hex_to_rgba(theme['bg'], 0.35)};
            min-height: 112px;
        }}

        .kpi-label {{
            color: {theme['text_secondary']};
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }}

        .kpi-value {{
            color: {theme['text_primary']};
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: 0.2px;
        }}

        .insight-card {{
            background: {hex_to_rgba(theme['card_bg'], 0.78)};
            border-left: 4px solid {theme['accent_blue']};
            border-radius: 14px;
            padding: 0.85rem 1rem;
            color: {theme['text_primary']};
            line-height: 1.45;
        }}

        .insight-mini {{
            color: {theme['text_secondary']};
            font-size: 0.9rem;
            margin-top: 0.25rem;
            margin-bottom: 0.95rem;
        }}

        .best-model-badge {{
            display: inline-block;
            background: {hex_to_rgba(theme['success_green'], 0.2)};
            border: 1px solid {hex_to_rgba(theme['success_green'], 0.65)};
            color: {theme['text_primary']};
            font-weight: 700;
            border-radius: 999px;
            padding: 0.3rem 0.7rem;
            font-size: 0.88rem;
        }}

        .stPlotlyChart {{
            border-radius: 12px;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="popover"] > div {{
            border-radius: 10px;
        }}

        .block-container {{
            padding-top: 5rem;
            padding-bottom: 1.2rem;
        }}

        @media (max-width: 768px) {{
            .block-container {{
                padding-top: 5.5rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_dashboard_data_dir(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "output" / "step7" / "dashboard_pack",
        project_root / "output" / "step6",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def find_raw_dataset_path(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "data" / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv",
        project_root / "data" / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018q4.csv",
        project_root / "data" / "accepted_2007_to_2018Q4.csv",
        project_root / "data" / "accepted_2007_to_2018q4.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def get_missing_files(base_dir: Path) -> List[str]:
    missing = []
    for filename in REQUIRED_FILES:
        if not (base_dir / filename).exists():
            missing.append(filename)
    return missing


def normalize_model_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def infer_model_name_from_feature_file(file_name: str, model_lookup: Dict[str, str]) -> str:
    token = Path(file_name).stem
    if token.startswith("feature_importance_"):
        token = token[len("feature_importance_") :]

    token_key = normalize_model_key(token)
    token_key = {
        "rf": "randomforestclassifier",
        "gbt": "gbtclassifier",
    }.get(token_key, token_key)

    if token_key in model_lookup:
        return model_lookup[token_key]

    for normalized, model_name in model_lookup.items():
        if token_key and (token_key in normalized or normalized in token_key):
            return model_name

    return token


def get_feature_importance_source_files(base_dir: Path) -> List[Path]:
    candidate_dirs = [base_dir]

    # If dashboard data is loaded from step7 pack, also pull richer model-level
    # feature importance exports from output/step6 when present.
    if base_dir.name == "dashboard_pack" and base_dir.parent.name == "step7":
        step6_dir = base_dir.parent.parent / "step6"
        if step6_dir.exists() and step6_dir.is_dir():
            candidate_dirs.append(step6_dir)

    resolved: Dict[str, Path] = {}
    for directory in candidate_dirs:
        for csv_path in sorted(directory.glob("feature_importance_*.csv")):
            # Prefer later directories for duplicate file names.
            resolved[csv_path.name] = csv_path

    return list(resolved.values())


@st.cache_data(show_spinner=False)
def load_dashboard_data(base_dir: str) -> Dict[str, pd.DataFrame]:
    root = Path(base_dir)
    data_frames: Dict[str, pd.DataFrame] = {}
    for filename in REQUIRED_FILES:
        data_frames[Path(filename).stem] = pd.read_csv(root / filename)

    fi_frames: List[pd.DataFrame] = []
    for csv_path in get_feature_importance_source_files(root):
        try:
            fi_df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "feature" not in fi_df.columns or "importance" not in fi_df.columns:
            continue

        fi_subset = fi_df[["feature", "importance"]].copy()
        fi_subset["source_file"] = csv_path.name
        fi_frames.append(fi_subset)

    if fi_frames:
        data_frames["feature_importance_all_raw"] = pd.concat(fi_frames, ignore_index=True)
    else:
        data_frames["feature_importance_all_raw"] = pd.DataFrame(columns=["feature", "importance", "source_file"])

    return data_frames


@st.cache_data(show_spinner=False)
def load_raw_eda_data(raw_path: str) -> pd.DataFrame:
    wanted_cols = [
        "loan_amnt",
        "annual_inc",
        "dti",
        "loan_status",
        "issue_d",
        "int_rate",
        "revol_util",
        "installment",
        "open_acc",
        "total_acc",
        "grade",
        "purpose",
        "home_ownership",
        "emp_length",
    ]

    header_df = pd.read_csv(raw_path, nrows=0)
    available_cols = [col for col in wanted_cols if col in header_df.columns]
    if not available_cols:
        return pd.DataFrame()

    # Keep raw EDA memory-safe in Streamlit by loading only a bounded sample.
    chunk_size = 50_000
    max_rows = 250_000

    def _collect_chunks(engine: Optional[str] = None) -> pd.DataFrame:
        read_kwargs: Dict[str, object] = {
            "usecols": available_cols,
            "chunksize": chunk_size,
            "low_memory": True,
            "on_bad_lines": "skip",
        }
        if engine:
            read_kwargs["engine"] = engine

        chunks: List[pd.DataFrame] = []
        loaded_rows = 0

        for chunk in pd.read_csv(raw_path, **read_kwargs):
            remaining = max_rows - loaded_rows
            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

            chunks.append(chunk)
            loaded_rows += len(chunk)

            if loaded_rows >= max_rows:
                break

        if not chunks:
            return pd.DataFrame(columns=available_cols)

        return pd.concat(chunks, ignore_index=True)

    try:
        return _collect_chunks()
    except pd.errors.ParserError:
        # Fall back to the Python engine for malformed/oversized CSV rows.
        return _collect_chunks(engine="python")


def to_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def prepare_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    model_comparison = data["model_comparison"].copy()
    to_numeric(model_comparison, ["accuracy", "f1", "precision", "recall", "auc", "tn", "fp", "fn", "tp"])

    metrics_long = data["model_metrics_long"].copy()
    to_numeric(metrics_long, ["value"])

    fi_gbt = data["feature_importance_gbt"].copy()
    to_numeric(fi_gbt, ["importance"])

    fi_rf = data["feature_importance_rf"].copy()
    to_numeric(fi_rf, ["importance"])

    fi_all_raw = data.get("feature_importance_all_raw", pd.DataFrame()).copy()
    if not fi_all_raw.empty:
        to_numeric(fi_all_raw, ["importance"])

    model_lookup = {
        normalize_model_key(model): str(model)
        for model in model_comparison["model"].dropna().astype(str).drop_duplicates().tolist()
    }

    fi_all_frames: List[pd.DataFrame] = []
    if not fi_all_raw.empty and {"feature", "importance", "source_file"}.issubset(fi_all_raw.columns):
        for source_file, source_df in fi_all_raw.groupby("source_file", dropna=False):
            mapped_model = infer_model_name_from_feature_file(str(source_file), model_lookup)
            feature_df = source_df[["feature", "importance"]].copy().dropna(subset=["feature", "importance"])
            if feature_df.empty:
                continue
            feature_df["model"] = mapped_model
            fi_all_frames.append(feature_df)

    if not fi_all_frames:
        fallback_sources = [
            ("GBTClassifier", fi_gbt),
            ("RandomForestClassifier", fi_rf),
        ]
        for model_name, source_df in fallback_sources:
            feature_df = source_df[["feature", "importance"]].copy().dropna(subset=["feature", "importance"])
            if feature_df.empty:
                continue
            feature_df["model"] = model_name
            fi_all_frames.append(feature_df)

    if fi_all_frames:
        fi_all = pd.concat(fi_all_frames, ignore_index=True)
        fi_all = (
            fi_all.groupby(["model", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["model", "importance"], ascending=[True, False])
            .reset_index(drop=True)
        )
        importance_sum = fi_all.groupby("model")["importance"].transform("sum")
        fi_all["importance_norm"] = np.where(importance_sum > 0, fi_all["importance"] / importance_sum, 0.0)
        fi_all["rank"] = fi_all.groupby("model").cumcount() + 1
    else:
        fi_all = pd.DataFrame(columns=["model", "feature", "importance", "importance_norm", "rank"])

    confusion = data["confusion_matrix_best"].copy()
    to_numeric(confusion, ["label", "prediction", "count"])

    roc = data["roc_curve_best"].copy()
    to_numeric(roc, ["fpr", "tpr", "threshold"])

    threshold = data["threshold_tuning_best_model"].copy()
    to_numeric(
        threshold,
        [
            "threshold",
            "tn",
            "fp",
            "fn",
            "tp",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "expected_cost",
            "fp_cost",
            "fn_cost",
        ],
    )

    cost = data["business_cost_matrix"].copy()
    to_numeric(
        cost,
        [
            "fp_cost",
            "fn_cost",
            "optimal_threshold",
            "minimum_expected_cost",
            "f1_at_optimal",
            "recall_at_optimal",
            "precision_at_optimal",
        ],
    )

    kpi = data["kpi_cards"].copy()

    loan_status = data["loan_status_distribution"].copy()
    to_numeric(loan_status, ["label", "count"])

    monthly = data["monthly_default_trend"].copy()
    to_numeric(monthly, ["default_rate", "loan_count"])
    if "issue_month" in monthly.columns:
        monthly["issue_month"] = pd.to_datetime(monthly["issue_month"], errors="coerce")

    fico = data["fico_default_risk"].copy()
    to_numeric(fico, ["fico_band", "default_rate", "loan_count"])

    home = data["home_ownership_risk"].copy()
    to_numeric(home, ["home_ownership_idx", "default_rate", "loan_count"])

    purpose = data["loan_purpose_risk"].copy()
    to_numeric(purpose, ["purpose_idx", "default_rate", "loan_count"])

    return {
        "model_comparison": model_comparison,
        "model_metrics_long": metrics_long,
        "feature_importance_gbt": fi_gbt,
        "feature_importance_rf": fi_rf,
        "feature_importance_all_models": fi_all,
        "confusion_matrix_best": confusion,
        "roc_curve_best": roc,
        "threshold_tuning_best_model": threshold,
        "business_cost_matrix": cost,
        "kpi_cards": kpi,
        "loan_status_distribution": loan_status,
        "monthly_default_trend": monthly,
        "fico_default_risk": fico,
        "home_ownership_risk": home,
        "loan_purpose_risk": purpose,
    }


def parse_simple_meta_yaml(file_path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not file_path.exists() or not file_path.is_file():
        return parsed

    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("\"").strip("'")
    return parsed


def load_mlflow_text_directory(directory: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not directory.exists() or not directory.is_dir():
        return values

    for child in sorted(directory.iterdir(), key=lambda p: p.name):
        if not child.is_file():
            continue
        values[child.name] = child.read_text(encoding="utf-8", errors="ignore").strip()
    return values


def parse_mlflow_metric_file(metric_path: Path) -> Tuple[Optional[float], Optional[int], Optional[pd.Timestamp]]:
    if not metric_path.exists() or not metric_path.is_file():
        return None, None, None

    lines = [line.strip() for line in metric_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return None, None, None

    parts = lines[-1].split()
    if len(parts) < 2:
        return None, None, None

    try:
        timestamp_raw = int(float(parts[0]))
        value = float(parts[1])
        step = int(float(parts[2])) if len(parts) > 2 else None
        timestamp = pd.to_datetime(timestamp_raw, unit="ms", errors="coerce")
    except (TypeError, ValueError):
        return None, None, None

    if pd.isna(timestamp):
        return value, step, None
    return value, step, timestamp


def parse_mlflow_model_history_tag(raw_text: str) -> Dict[str, object]:
    if not raw_text:
        return {}

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}

    if isinstance(payload, list) and payload:
        record = payload[-1]
    elif isinstance(payload, dict):
        record = payload
    else:
        return {}

    if not isinstance(record, dict):
        return {}

    flavors_raw = record.get("flavors", {})
    flavors = flavors_raw if isinstance(flavors_raw, dict) else {}
    flavor_names = ", ".join(sorted([str(key) for key in flavors.keys()])) if flavors else ""

    model_class = ""
    spark_flavor = flavors.get("spark")
    if isinstance(spark_flavor, dict):
        model_class = str(spark_flavor.get("model_class", "")).strip()

    return {
        "artifact_path": str(record.get("artifact_path", "")).strip(),
        "created_utc": str(record.get("utc_time_created", "")).strip(),
        "flavors": flavor_names,
        "model_class": model_class,
        "model_size_bytes": record.get("model_size_bytes", np.nan),
    }


@st.cache_data(show_spinner=False)
def load_mlflow_tracking_data(mlruns_root: str) -> Dict[str, pd.DataFrame]:
    run_rows: List[Dict[str, object]] = []
    param_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    tag_rows: List[Dict[str, object]] = []
    model_artifact_rows: List[Dict[str, object]] = []

    root = Path(mlruns_root)
    if not root.exists() or not root.is_dir():
        return {
            "runs": pd.DataFrame(),
            "params": pd.DataFrame(),
            "metrics": pd.DataFrame(),
            "tags": pd.DataFrame(),
            "model_artifacts": pd.DataFrame(),
        }

    ignored_dirs = {".trash", "models"}
    status_map = {
        "1": "RUNNING",
        "2": "SCHEDULED",
        "3": "FINISHED",
        "4": "FAILED",
        "5": "KILLED",
    }

    for experiment_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not experiment_dir.is_dir() or experiment_dir.name in ignored_dirs:
            continue

        exp_meta = parse_simple_meta_yaml(experiment_dir / "meta.yaml")
        experiment_id = exp_meta.get("experiment_id", experiment_dir.name)
        experiment_name = exp_meta.get("name", experiment_dir.name)

        for run_dir in sorted(experiment_dir.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue

            run_meta_file = run_dir / "meta.yaml"
            if not run_meta_file.exists():
                continue

            run_meta = parse_simple_meta_yaml(run_meta_file)
            run_id = str(run_meta.get("run_id", run_meta.get("run_uuid", run_dir.name))).strip()
            run_name = str(run_meta.get("run_name", "")).strip() or run_id[:8]

            raw_status = str(run_meta.get("status", "")).strip()
            status = status_map.get(raw_status, raw_status or "UNKNOWN")

            try:
                start_time = pd.to_datetime(int(float(run_meta.get("start_time", ""))), unit="ms", errors="coerce")
            except (TypeError, ValueError):
                start_time = pd.NaT

            try:
                end_time = pd.to_datetime(int(float(run_meta.get("end_time", ""))), unit="ms", errors="coerce")
            except (TypeError, ValueError):
                end_time = pd.NaT

            if pd.notna(start_time) and pd.notna(end_time):
                duration_seconds: Optional[float] = max((end_time - start_time).total_seconds(), 0.0)
            else:
                duration_seconds = None

            params = load_mlflow_text_directory(run_dir / "params")
            tags = load_mlflow_text_directory(run_dir / "tags")

            metrics: Dict[str, float] = {}
            metrics_dir = run_dir / "metrics"
            if metrics_dir.exists() and metrics_dir.is_dir():
                for metric_file in sorted(metrics_dir.iterdir(), key=lambda p: p.name):
                    if not metric_file.is_file():
                        continue
                    metric_value, metric_step, metric_timestamp = parse_mlflow_metric_file(metric_file)
                    if metric_value is None:
                        continue
                    metrics[metric_file.name] = metric_value
                    metric_rows.append(
                        {
                            "experiment_id": experiment_id,
                            "experiment_name": experiment_name,
                            "run_id": run_id,
                            "run_name": run_name,
                            "model_name": params.get("model_name", run_name),
                            "metric": metric_file.name,
                            "value": metric_value,
                            "step": metric_step,
                            "timestamp": metric_timestamp,
                        }
                    )

            for key, value in params.items():
                param_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment_name": experiment_name,
                        "run_id": run_id,
                        "run_name": run_name,
                        "model_name": params.get("model_name", run_name),
                        "key": key,
                        "value": value,
                    }
                )

            for key, value in tags.items():
                tag_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment_name": experiment_name,
                        "run_id": run_id,
                        "run_name": run_name,
                        "model_name": params.get("model_name", run_name),
                        "key": key,
                        "value": value,
                    }
                )

            model_info = parse_mlflow_model_history_tag(tags.get("mlflow.log-model.history", ""))
            has_model_artifact = bool(model_info)
            if not has_model_artifact:
                has_model_artifact = (run_dir / "artifacts" / "model" / "MLmodel").exists()

            if has_model_artifact:
                model_artifact_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment_name": experiment_name,
                        "run_id": run_id,
                        "run_name": run_name,
                        "model_name": params.get("model_name", run_name),
                        "artifact_path": model_info.get("artifact_path", "model"),
                        "created_utc": model_info.get("created_utc", ""),
                        "flavors": model_info.get("flavors", ""),
                        "model_class": model_info.get("model_class", ""),
                        "model_size_bytes": model_info.get("model_size_bytes", np.nan),
                    }
                )

            model_name = str(params.get("model_name", run_name)).strip() or run_name
            has_params = len(params) > 0
            has_metrics = len(metrics) > 0
            fully_logged = has_params and has_metrics and has_model_artifact

            run_rows.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                    "run_id": run_id,
                    "run_name": run_name,
                    "model_name": model_name,
                    "status": status,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration_seconds,
                    "has_params": has_params,
                    "has_metrics": has_metrics,
                    "has_model_artifact": has_model_artifact,
                    "fully_logged": fully_logged,
                    "accuracy": metrics.get("accuracy", np.nan),
                    "f1": metrics.get("f1", np.nan),
                    "precision": metrics.get("precision", np.nan),
                    "recall": metrics.get("recall", np.nan),
                    "auc": metrics.get("auc", np.nan),
                    "param_count": len(params),
                    "metric_count": len(metrics),
                }
            )

    runs_df = pd.DataFrame(run_rows)
    if not runs_df.empty:
        runs_df = runs_df.sort_values("start_time", ascending=False, na_position="last")

    params_df = pd.DataFrame(param_rows)
    metrics_df = pd.DataFrame(metric_rows)
    tags_df = pd.DataFrame(tag_rows)
    model_artifacts_df = pd.DataFrame(model_artifact_rows)

    return {
        "runs": runs_df,
        "params": params_df,
        "metrics": metrics_df,
        "tags": tags_df,
        "model_artifacts": model_artifacts_df,
    }


def build_feature_importance_data(
    prepared: Dict[str, pd.DataFrame],
    selected_models: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fi_all = prepared.get("feature_importance_all_models", pd.DataFrame()).copy()

    model_list = (
        prepared["model_comparison"]["model"].dropna().astype(str).drop_duplicates().tolist()
        if "model_comparison" in prepared and "model" in prepared["model_comparison"].columns
        else []
    )
    if not model_list and not fi_all.empty and "model" in fi_all.columns:
        model_list = sorted(fi_all["model"].dropna().astype(str).unique().tolist())

    if selected_models:
        selected_set = set(selected_models)
        model_list = [model for model in model_list if model in selected_set]

    importance_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []

    for model_name in model_list:
        source_df = fi_all.loc[fi_all["model"] == model_name].copy() if not fi_all.empty else pd.DataFrame()

        if source_df.empty or "feature" not in source_df.columns or "importance" not in source_df.columns:
            coverage_rows.append(
                {
                    "model": model_name,
                    "importance_available": False,
                    "feature_count": 0,
                    "top_feature": "N/A",
                    "top_importance": np.nan,
                }
            )
            continue

        fi_df = source_df[["feature", "importance"]].copy()
        fi_df["importance"] = pd.to_numeric(fi_df["importance"], errors="coerce")
        fi_df = fi_df.dropna(subset=["feature", "importance"])

        if fi_df.empty:
            coverage_rows.append(
                {
                    "model": model_name,
                    "importance_available": False,
                    "feature_count": 0,
                    "top_feature": "N/A",
                    "top_importance": np.nan,
                }
            )
            continue

        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
        importance_sum = float(fi_df["importance"].sum())
        fi_df["importance_norm"] = fi_df["importance"] / importance_sum if importance_sum > 0 else 0.0
        fi_df["model"] = model_name
        fi_df["rank"] = np.arange(1, len(fi_df) + 1)

        top_row = fi_df.iloc[0]
        coverage_rows.append(
            {
                "model": model_name,
                "importance_available": True,
                "feature_count": int(len(fi_df)),
                "top_feature": str(top_row["feature"]),
                "top_importance": float(top_row["importance"]),
            }
        )

        importance_rows.extend(fi_df.to_dict("records"))

    importance_long = pd.DataFrame(importance_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    return importance_long, coverage_df


def clip_series(series: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    lower = numeric.quantile(q_low)
    upper = numeric.quantile(q_high)
    return numeric.clip(lower, upper)


def prepare_raw_eda(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    eda = raw_df.copy()

    for base_col in ["loan_amnt", "annual_inc", "dti", "installment", "open_acc", "total_acc"]:
        if base_col in eda.columns:
            eda[base_col] = pd.to_numeric(eda[base_col], errors="coerce")

    for pct_col in ["int_rate", "revol_util"]:
        if pct_col in eda.columns:
            eda[f"{pct_col}_num"] = pd.to_numeric(
                eda[pct_col].astype(str).str.replace("%", "", regex=False), errors="coerce"
            )

    if "issue_d" in eda.columns:
        issue_date = pd.to_datetime(eda["issue_d"], format="%b-%Y", errors="coerce")
        if issue_date.isna().all():
            issue_date = pd.to_datetime(eda["issue_d"], errors="coerce")
        eda["issue_date"] = issue_date

    if "loan_status" in eda.columns:
        status_lower = eda["loan_status"].astype(str).str.lower()
        eda["default_flag"] = status_lower.str.contains("charged off|default|late", regex=True, na=False).astype(int)
        eda["default_status"] = np.where(eda["default_flag"] == 1, "Default", "Non-Default")

    for col in ["loan_amnt", "annual_inc", "dti"]:
        if col in eda.columns:
            eda[f"{col}_clip"] = clip_series(eda[col])

    return eda


def format_value(value: object, kind: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if kind == "int":
        return f"{numeric:,.0f}"
    if kind == "pct":
        return f"{numeric * 100:,.2f}%"
    if kind == "auc":
        return f"{numeric:,.4f}"
    if kind == "rate":
        return f"{numeric:,.2f}%"
    if kind == "threshold":
        return f"{numeric:,.2f}"
    return f"{numeric:,.2f}"


def get_kpi_map(kpi_df: pd.DataFrame) -> Dict[str, object]:
    kpi_map: Dict[str, object] = {}
    if kpi_df.empty:
        return kpi_map

    for _, row in kpi_df.iterrows():
        key = str(row.get("kpi", "")).strip()
        val = row.get("value", np.nan)
        kpi_map[key] = val
    return kpi_map


def get_best_model_row(model_comparison: pd.DataFrame) -> pd.Series:
    if model_comparison.empty:
        return pd.Series(dtype="object")
    return model_comparison.sort_values("auc", ascending=False).iloc[0]


def style_figure(fig: go.Figure, theme: Dict[str, str], height: int = 390) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=hex_to_rgba(theme["card_bg"], 0.45),
        font=dict(color=theme["text_primary"], size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=35, r=20, t=55, b=35),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor=hex_to_rgba(theme["grid"], 0.45), zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=hex_to_rgba(theme["grid"], 0.45), zeroline=False)
    return fig


def render_kpi_cards(kpi_map: Dict[str, object], theme: Dict[str, str]) -> None:
    specs = [
        ("Total Loans", "int"),
        ("Default Rate", "pct"),
        ("Best Model", "text"),
        ("Best AUC", "auc"),
        ("Avg Interest Rate", "rate"),
    ]

    cols = st.columns(5, gap="medium")
    for col, (kpi_name, value_type) in zip(cols, specs):
        raw_value = kpi_map.get(kpi_name, "N/A")
        if value_type == "text":
            formatted = str(raw_value)
        else:
            formatted = format_value(raw_value, value_type)

        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{kpi_name}</div>
                    <div class="kpi-value">{formatted}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_eda_notes(insight_text: str, axis_text: str) -> None:
    st.markdown(
        f'<div class="insight-mini"><strong>Yorum:</strong> {insight_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="insight-mini" style="margin-top:-0.65rem;"><strong>Eksen açıklaması:</strong> {axis_text}</div>',
        unsafe_allow_html=True,
    )


def risk_profile_text(
    fico_df: pd.DataFrame,
    home_df: pd.DataFrame,
    purpose_df: pd.DataFrame,
    min_volume: int = 1000,
) -> str:
    chunks = []

    fico_candidate = fico_df.loc[fico_df["loan_count"] >= min_volume] if not fico_df.empty else pd.DataFrame()
    if not fico_candidate.empty:
        top_fico = fico_candidate.sort_values("default_rate", ascending=False).iloc[0]
        chunks.append(
            f"FICO bandı {int(top_fico['fico_band'])} seviyesinde temerrüt riski %{top_fico['default_rate'] * 100:.1f} düzeyine ulaşıyor."
        )

    home_candidate = home_df.loc[home_df["loan_count"] >= min_volume] if not home_df.empty else pd.DataFrame()
    if not home_candidate.empty:
        top_home = home_candidate.sort_values("default_rate", ascending=False).iloc[0]
        chunks.append(
            f"Konut sahipliği segmenti {int(top_home['home_ownership_idx'])} için risk %{top_home['default_rate'] * 100:.1f} seviyesinde yüksek seyrediyor."
        )

    purpose_candidate = purpose_df.loc[purpose_df["loan_count"] >= min_volume] if not purpose_df.empty else pd.DataFrame()
    if not purpose_candidate.empty:
        top_purpose = purpose_candidate.sort_values("default_rate", ascending=False).iloc[0]
        chunks.append(
            f"Amaç segmenti {int(top_purpose['purpose_idx'])}, yeterli hacimde en yüksek riskli davranış kümesi olarak öne çıkıyor."
        )

    if not chunks:
        return "Düşük FICO ve yüksek DTI değerine sahip müşterilerde temerrüt riski belirgin şekilde artıyor."

    return " ".join(chunks)


def render_executive_overview(
    prepared: Dict[str, pd.DataFrame],
    kpi_map: Dict[str, object],
    monthly_filtered: pd.DataFrame,
    theme: Dict[str, str],
) -> None:
    st.markdown('<div class="main-title">Credit Risk Analysis & Machine Learning Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Portföy riski, model kalitesi ve stratejik eşik kararlarının üst düzey özeti.</div>',
        unsafe_allow_html=True,
    )

    render_kpi_cards(kpi_map, theme)
    st.write("")

    chart_col1, chart_col2, insight_col = st.columns([1.0, 1.4, 1.1], gap="large")

    with chart_col1:
        st.subheader("Loan Status Distribution")
        status_df = prepared["loan_status_distribution"].copy()
        if "segment" not in status_df.columns:
            status_df["segment"] = status_df["label"].map({0: "Non-Default", 1: "Default"}).fillna("Unknown")

        pie_fig = px.pie(
            status_df,
            names="segment",
            values="count",
            hole=0.5,
            color="segment",
            color_discrete_map={
                "Default": theme["risk_red"],
                "Non-Default": theme["success_green"],
                "Unknown": theme["warning_orange"],
            },
        )
        pie_fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>Count=%{value:,}<extra></extra>")
        style_figure(pie_fig, theme, height=390)
        st.plotly_chart(pie_fig, use_container_width=True, config={"displaylogo": False})
        st.markdown('<div class="insight-mini">Kredilerin çoğu temerrüde düşmemiş olsa da temerrüt segmentinin büyüklüğü zarar tahminlerini anlamlı biçimde etkiliyor.</div>', unsafe_allow_html=True)

    with chart_col2:
        st.subheader("Monthly Default Trend")
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=monthly_filtered["issue_month"],
                y=monthly_filtered["default_rate"],
                mode="lines+markers",
                name="Default Rate",
                line=dict(color=theme["accent_blue"], width=3),
                marker=dict(size=6),
                hovertemplate="%{x|%Y-%m}<br>Default Rate=%{y:.2%}<extra></extra>",
            )
        )
        trend_fig.update_yaxes(tickformat=".0%")
        style_figure(trend_fig, theme, height=390)
        st.plotly_chart(trend_fig, use_container_width=True, config={"displaylogo": False})
        st.markdown('<div class="insight-mini">Temerrüt davranışı zaman içinde değişiyor; bu nedenle sabit politika yerine dinamik risk ayarı daha etkili oluyor.</div>', unsafe_allow_html=True)

    with insight_col:
        st.subheader("Risk Profile Insight")
        text = risk_profile_text(
            prepared["fico_default_risk"],
            prepared["home_ownership_risk"],
            prepared["loan_purpose_risk"],
        )
        st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)

        rec_cost = format_value(kpi_map.get("Recommended Threshold (Cost)", np.nan), "threshold")
        rec_f1 = format_value(kpi_map.get("Recommended Threshold (F1)", np.nan), "threshold")

        st.write("")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Recommended Thresholds</div>
                <div class="kpi-value">Cost: {rec_cost}</div>
                <div class="kpi-value" style="font-size:1.1rem; margin-top:0.25rem;">F1: {rec_f1}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_eda_dashboard(
    prepared: Dict[str, pd.DataFrame],
    raw_eda: pd.DataFrame,
    monthly_filtered: pd.DataFrame,
    theme: Dict[str, str],
) -> None:
    st.markdown('<div class="main-title">EDA Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Dağılım analizi, davranışsal risk teşhisi ve portföy kalite sinyalleri.</div>',
        unsafe_allow_html=True,
    )

    if raw_eda.empty:
        st.warning("Raw EDA dataset could not be loaded. Histograms, missing values, and correlation heatmap are unavailable.")

    hist_cols = st.columns(3, gap="medium")
    hist_targets = [
        (
            "loan_amnt_clip",
            "Loan Amount Distribution",
            "Kredi tutarı yoğunlaşması, portföyde baskın bilet büyüklüklerini ve tahsis maruziyetini gösteriyor.",
            "X ekseni kredi tutarını (uç değerleri kırpılmış), Y ekseni bu aralıktaki kredi adedini gösterir.",
        ),
        (
            "annual_inc_clip",
            "Annual Income Distribution",
            "Gelir dağılımındaki çarpıklık, model eğitiminden önce güçlü normalizasyon ihtiyacına işaret ediyor.",
            "X ekseni yıllık geliri (uç değerleri kırpılmış), Y ekseni ilgili gelir aralığındaki müşteri sayısını gösterir.",
        ),
        (
            "dti_clip",
            "DTI Distribution",
            "Yüksek DTI kuyrukları, finansal olarak zorlanabilecek müşteri yoğunluğunu gösteriyor.",
            "X ekseni DTI (borç/gelir oranı), Y ekseni bu DTI aralığındaki gözlem sayısını ifade eder.",
        ),
    ]

    for col, (field, title, insight, axis_note) in zip(hist_cols, hist_targets):
        with col:
            st.subheader(title)
            if raw_eda.empty or field not in raw_eda.columns:
                st.info("Data unavailable")
            else:
                fig = px.histogram(
                    raw_eda,
                    x=field,
                    nbins=60,
                    color_discrete_sequence=[theme["accent_blue"]],
                    opacity=0.9,
                )
                fig.update_traces(hovertemplate="Value=%{x:.2f}<br>Count=%{y:,}<extra></extra>")
                style_figure(fig, theme, height=320)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                render_eda_notes(insight, axis_note)

    row2_col1, row2_col2 = st.columns([1, 1.7], gap="large")

    with row2_col1:
        st.subheader("Loan Status Pie")
        status_df = prepared["loan_status_distribution"].copy()
        if "segment" not in status_df.columns:
            status_df["segment"] = status_df["label"].map({0: "Non-Default", 1: "Default"}).fillna("Unknown")

        fig = px.pie(
            status_df,
            names="segment",
            values="count",
            hole=0.45,
            color="segment",
            color_discrete_map={
                "Default": theme["risk_red"],
                "Non-Default": theme["success_green"],
                "Unknown": theme["warning_orange"],
            },
        )
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}<br>Count=%{value:,}<extra></extra>")
        style_figure(fig, theme, height=330)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        render_eda_notes(
            "Temerrüt oranı hala anlamlı düzeyde yüksek; bu durum hedefli risk segmentasyonu ihtiyacını destekliyor.",
            ".",
        )

    with row2_col2:
        st.subheader("Monthly Default Trend")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly_filtered["issue_month"],
                y=monthly_filtered["default_rate"],
                mode="lines+markers",
                line=dict(color=theme["accent_blue"], width=3),
                marker=dict(size=6),
                name="Default Rate",
                hovertemplate="%{x|%Y-%m}<br>Default Rate=%{y:.2%}<extra></extra>",
            )
        )
        fig.update_yaxes(tickformat=".0%")
        style_figure(fig, theme, height=330)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        render_eda_notes(
            "Zamansal dalgalanmalar, geri ödeme davranışının makro döngülere duyarlı olduğunu gösteriyor.",
            "X ekseni ay bazında dönem bilgisini, Y ekseni ilgili aydaki temerrüt oranını (%) gösterir.",
        )

    row3 = st.columns(3, gap="medium")

    with row3[0]:
        st.subheader("FICO Risk Analysis")
        fico = prepared["fico_default_risk"].sort_values("fico_band")
        fig = px.bar(
            fico,
            x="fico_band",
            y="default_rate",
            color="default_rate",
            color_continuous_scale=[theme["accent_blue"], theme["risk_red"]],
        )
        fig.update_traces(hovertemplate="FICO Band=%{x}<br>Default Rate=%{y:.2%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%")
        style_figure(fig, theme, height=330)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        render_eda_notes(
            "Daha düşük kredi kalitesi bantlarında temerrüt oranı belirgin şekilde daha yüksek seyrediyor.",
            "X ekseni FICO bandını, Y ekseni her banttaki temerrüt oranını (%) ifade eder.",
        )

    with row3[1]:
        st.subheader("Home Ownership Risk")
        home = prepared["home_ownership_risk"].copy().sort_values("default_rate", ascending=False)
        home["segment"] = home["home_ownership_idx"].fillna(-1).astype(int).astype(str)
        fig = px.bar(
            home,
            x="segment",
            y="default_rate",
            color="default_rate",
            color_continuous_scale=[theme["accent_blue"], theme["risk_red"]],
        )
        fig.update_traces(hovertemplate="Home Segment=%{x}<br>Default Rate=%{y:.2%}<br>Loan Count=%{customdata[0]:,}<extra></extra>", customdata=home[["loan_count"]])
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(title_text="home_ownership_idx")
        style_figure(fig, theme, height=330)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        render_eda_notes(
            "Konut sahipliği segmentleri arasında anlamlı risk farkları var; fiyatlama politikasında dikkate alınmalı.",
            "X ekseni konut sahipliği segment indeksini, Y ekseni segmentin temerrüt oranını (%) gösterir.",
        )

    with row3[2]:
        st.subheader("Loan Purpose Risk")
        purpose = prepared["loan_purpose_risk"].copy()
        purpose = purpose.loc[purpose["loan_count"] >= 1000].sort_values("default_rate", ascending=False).head(10)
        purpose["segment"] = purpose["purpose_idx"].fillna(-1).astype(int).astype(str)
        fig = px.bar(
            purpose,
            x="segment",
            y="default_rate",
            color="default_rate",
            color_continuous_scale=[theme["accent_blue"], theme["risk_red"]],
        )
        fig.update_traces(hovertemplate="Purpose Segment=%{x}<br>Default Rate=%{y:.2%}<br>Loan Count=%{customdata[0]:,}<extra></extra>", customdata=purpose[["loan_count"]])
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(title_text="purpose_idx")
        style_figure(fig, theme, height=330)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        render_eda_notes(
            "Temerrüt oranı yüksek amaç segmentleri daha sıkı politika kurallarıyla yönetilebilir.",
            "X ekseni kredi amaç segment indeksini, Y ekseni ilgili segmentin temerrüt oranını (%) ifade eder.",
        )

    row4_col1, row4_col2 = st.columns(2, gap="large")

    with row4_col1:
        st.subheader("Missing Value Summary")
        if raw_eda.empty:
            st.info("Data unavailable")
        else:
            missing = (raw_eda.isna().mean() * 100).sort_values(ascending=False)
            missing = missing[missing > 0].head(12).reset_index()
            missing.columns = ["column", "missing_pct"]
            if missing.empty:
                st.success("No missing values in selected EDA fields.")
            else:
                fig = px.bar(
                    missing,
                    x="missing_pct",
                    y="column",
                    orientation="h",
                    color="missing_pct",
                    color_continuous_scale=[theme["accent_blue"], theme["warning_orange"], theme["risk_red"]],
                )
                fig.update_traces(hovertemplate="%{y}<br>Missing=%{x:.2f}%<extra></extra>")
                fig.update_xaxes(title_text="Missing %")
                style_figure(fig, theme, height=350)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                render_eda_notes(
                    "Eksik oranı yüksek alanlar için imputasyon stratejisi veya dışlama kuralı gerekebilir.",
                    "X ekseni eksik değer yüzdesini, Y ekseni ilgili sütun adını gösterir.",
                )

    with row4_col2:
        st.subheader("Correlation Heatmap")
        if raw_eda.empty:
            st.info("Data unavailable")
        else:
            corr_fields = [
                "loan_amnt",
                "annual_inc",
                "dti",
                "installment",
                "int_rate_num",
                "revol_util_num",
                "open_acc",
                "total_acc",
                "default_flag",
            ]
            corr_fields = [field for field in corr_fields if field in raw_eda.columns]
            if len(corr_fields) < 2:
                st.info("Insufficient numeric fields for correlation.")
            else:
                corr = raw_eda[corr_fields].corr(numeric_only=True)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        zmin=-1,
                        zmax=1,
                        colorscale=[
                            [0.0, theme["risk_red"]],
                            [0.5, theme["card_bg"]],
                            [1.0, theme["accent_blue"]],
                        ],
                        hovertemplate="%{y} vs %{x}<br>Correlation=%{z:.2f}<extra></extra>",
                    )
                )
                style_figure(fig, theme, height=350)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                render_eda_notes(
                    "Yüksek kullanım oranı ve borç yükü, çoğunlukla artan temerrüt davranışıyla birlikte hareket ediyor.",
                    "X ve Y eksenlerinde sayısal değişkenler yer alır; hücre rengi (Z değeri) iki değişken arasındaki korelasyon katsayısını gösterir.",
                )

    st.markdown("### Additional EDA Visuals")
    extra_col1, extra_col2, extra_col3 = st.columns(3, gap="medium")

    with extra_col1:
        st.subheader("DTI by Default Status")
        if raw_eda.empty or "dti_clip" not in raw_eda.columns or "default_status" not in raw_eda.columns:
            st.info("Data unavailable")
        else:
            box_df = raw_eda[["dti_clip", "default_status"]].dropna().sample(min(50000, len(raw_eda)), random_state=42)
            fig = px.box(
                box_df,
                x="default_status",
                y="dti_clip",
                color="default_status",
                color_discrete_map={"Default": theme["risk_red"], "Non-Default": theme["success_green"]},
            )
            fig.update_traces(hovertemplate="Status=%{x}<br>DTI=%{y:.2f}<extra></extra>")
            style_figure(fig, theme, height=320)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            render_eda_notes(
                "Temerrüt grubunda medyan DTI daha yüksek ve risk dağılımı daha geniş görünüyor.",
                "X ekseni temerrüt durumunu (Default/Non-Default), Y ekseni DTI değerini gösterir.",
            )

    with extra_col2:
        st.subheader("Income vs Loan Amount")
        if raw_eda.empty or "annual_inc_clip" not in raw_eda.columns or "loan_amnt_clip" not in raw_eda.columns:
            st.info("Data unavailable")
        else:
            scatter_fields = ["annual_inc_clip", "loan_amnt_clip"]
            if "default_status" in raw_eda.columns:
                scatter_fields.append("default_status")
            scatter_df = raw_eda[scatter_fields].dropna().sample(min(30000, len(raw_eda)), random_state=42)
            fig = px.scatter(
                scatter_df,
                x="annual_inc_clip",
                y="loan_amnt_clip",
                color="default_status" if "default_status" in scatter_df.columns else None,
                color_discrete_map={"Default": theme["risk_red"], "Non-Default": theme["accent_blue"]},
                opacity=0.55,
            )
            fig.update_traces(hovertemplate="Income=%{x:,.0f}<br>Loan=%{y:,.0f}<extra></extra>")
            style_figure(fig, theme, height=320)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            render_eda_notes(
                "Düşük gelir ve yüksek kredi kombinasyonları, daha riskli bölgelerde kümelenme eğilimi gösteriyor.",
                "X ekseni yıllık gelir seviyesini, Y ekseni kredi tutarını; renk ise temerrüt durumunu ifade eder.",
            )

    with extra_col3:
        st.subheader("Interest Rate by Default")
        if raw_eda.empty or "int_rate_num" not in raw_eda.columns or "default_status" not in raw_eda.columns:
            st.info("Data unavailable")
        else:
            hist_df = raw_eda[["int_rate_num", "default_status"]].dropna().sample(min(80000, len(raw_eda)), random_state=42)
            fig = px.histogram(
                hist_df,
                x="int_rate_num",
                color="default_status",
                barmode="overlay",
                nbins=60,
                opacity=0.55,
                color_discrete_map={"Default": theme["risk_red"], "Non-Default": theme["accent_blue"]},
            )
            fig.update_traces(hovertemplate="Rate=%{x:.2f}<br>Count=%{y:,}<extra></extra>")
            style_figure(fig, theme, height=320)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            render_eda_notes(
                "Yüksek faiz oranı bantları, gerçekleşen temerrüt riskindeki artışla ilişkili görünüyor.",
                "X ekseni faiz oranı aralığını, Y ekseni gözlem sayısını; renk ise temerrüt durumunu gösterir.",
            )


def render_ml_dashboard(
    prepared: Dict[str, pd.DataFrame],
    selected_models: List[str],
    kpi_map: Dict[str, object],
    theme: Dict[str, str],
) -> None:
    st.markdown('<div class="main-title">Machine Learning Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Karşılaştırmalı model analizi, açıklanabilirlik ve sınıflandırma kalitesi.</div>',
        unsafe_allow_html=True,
    )

    best_model = str(kpi_map.get("Best Model", "GBTClassifier"))
    best_auc = format_value(kpi_map.get("Best AUC", np.nan), "auc")

    st.markdown(
        f"""
        <div class="kpi-card" style="margin-bottom: 1rem;">
            <div class="kpi-label">Best Model Highlight</div>
            <div class="kpi-value">{best_model} <span class="best-model-badge">Primary Production Candidate</span></div>
            <div class="insight-mini" style="margin-bottom:0;">En iyi AUC: {best_auc}. GBTClassifier kıyaslama için referans model olarak öne çıkarılıyor.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_long = prepared["model_metrics_long"].copy()
    if selected_models:
        metrics_long = metrics_long.loc[metrics_long["model"].isin(selected_models)]

    metrics_order = ["accuracy", "f1", "precision", "recall", "auc"]
    metrics_long["metric"] = pd.Categorical(metrics_long["metric"], categories=metrics_order, ordered=True)

    st.subheader("Model Metrics Comparison")
    grouped_bar = px.bar(
        metrics_long.sort_values(["metric", "model"]),
        x="metric",
        y="value",
        color="model",
        barmode="group",
        color_discrete_sequence=[
            theme["accent_blue"],
            theme["success_green"],
            theme["warning_orange"],
            theme["risk_red"],
            "#8AA4C8",
        ],
        hover_data={"value": ":.4f"},
    )
    grouped_bar.update_traces(hovertemplate="Model=%{fullData.name}<br>Metric=%{x}<br>Value=%{y:.4f}<extra></extra>")
    grouped_bar.update_yaxes(range=[0, max(0.8, float(metrics_long["value"].max()) + 0.05)])
    style_figure(grouped_bar, theme, height=420)
    st.plotly_chart(grouped_bar, use_container_width=True, config={"displaylogo": False})
    st.markdown('<div class="insight-mini">Model sıralamasında AUC önceliklidir; GBTClassifier en güçlü ayırt etme performansını koruyor.</div>', unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2, gap="large")

    with row2_col1:
        st.subheader("Feature Importance (Horizontal Bar Chart)")
        fi_all = prepared.get("feature_importance_all_models", pd.DataFrame()).copy()
        if selected_models:
            fi_all = fi_all.loc[fi_all["model"].isin(selected_models)]

        available_models = sorted(fi_all["model"].dropna().astype(str).unique().tolist()) if not fi_all.empty else []

        if not available_models:
            st.info("Seçili model filtresinde feature importance verisi bulunamadı.")
        else:
            selected_importance_model = st.selectbox(
                "Feature Importance Model",
                options=available_models,
                index=0,
                key="feature_importance_model_select",
            )

            model_feature_count = int(fi_all.loc[fi_all["model"] == selected_importance_model, "feature"].nunique())
            min_features = 1 if model_feature_count < 5 else 5
            max_features = max(min_features, min(40, model_feature_count))

            top_n_features = st.slider(
                "Gösterilecek özellik sayısı",
                min_value=min_features,
                max_value=max_features,
                value=min(15, max_features),
                step=1,
                key="feature_importance_top_n",
            )

            fi_df = (
                fi_all.loc[fi_all["model"] == selected_importance_model]
                .sort_values("importance", ascending=False)
                .head(top_n_features)
                .sort_values("importance", ascending=True)
            )

            fi_fig = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance_norm",
                color_continuous_scale=[theme["accent_blue"], theme["risk_red"]],
            )
            fi_fig.update_traces(hovertemplate="Feature=%{y}<br>Importance=%{x:.4f}<extra></extra>")
            style_figure(fi_fig, theme, height=420)
            st.plotly_chart(fi_fig, use_container_width=True, config={"displaylogo": False})
            st.markdown('<div class="insight-mini">Model bazında en etkili değişkenler, tahmin davranışının açıklanabilirliğini güçlendirir.</div>', unsafe_allow_html=True)
            st.markdown('<div class="insight-mini" style="margin-top:-0.65rem;"><strong>Eksen açıklaması:</strong> X ekseni özellik önem skorunu (importance), Y ekseni modelde kullanılan özellik adlarını gösterir.</div>', unsafe_allow_html=True)

    with row2_col2:
        st.subheader("ROC Curve")
        roc = prepared["roc_curve_best"].copy().dropna(subset=["fpr", "tpr"]).sort_values("fpr")

        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                name="GBTClassifier ROC",
                line=dict(color=theme["accent_blue"], width=3),
                hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Baseline",
                line=dict(color=theme["text_secondary"], width=2, dash="dash"),
                hoverinfo="skip",
            )
        )
        roc_fig.update_xaxes(range=[0, 1], title_text="False Positive Rate")
        roc_fig.update_yaxes(range=[0, 1], title_text="True Positive Rate")
        style_figure(roc_fig, theme, height=420)
        st.plotly_chart(roc_fig, use_container_width=True, config={"displaylogo": False})
        st.markdown('<div class="insight-mini">Eğri formu, sınıflar arasında anlamlı ayrışma ve istikrarlı sıralama kabiliyetini doğruluyor.</div>', unsafe_allow_html=True)

    row3_col1, row3_col2 = st.columns([1.15, 1.0], gap="large")

    with row3_col1:
        st.subheader("Confusion Matrix Heatmap")
        cm = prepared["confusion_matrix_best"].copy()
        cm_pivot = cm.pivot_table(index="label", columns="prediction", values="count", aggfunc="sum").fillna(0)
        cm_pivot = cm_pivot.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm_pivot.values,
                x=["Pred 0 (Non-Default)", "Pred 1 (Default)"],
                y=["Actual 0 (Non-Default)", "Actual 1 (Default)"],
                colorscale=[
                    [0.0, hex_to_rgba(theme["accent_blue"], 0.2)],
                    [0.6, theme["accent_blue"]],
                    [1.0, theme["risk_red"]],
                ],
                hovertemplate="%{y}<br>%{x}<br>Count=%{z:,.0f}<extra></extra>",
            )
        )
        style_figure(cm_fig, theme, height=390)
        st.plotly_chart(cm_fig, use_container_width=True, config={"displaylogo": False})
        st.markdown('<div class="insight-mini">Yanlış negatifler, yüksek iş etkisi maliyeti nedeniyle stratejik olarak kritik önem taşıyor.</div>', unsafe_allow_html=True)

    with row3_col2:
        st.subheader("Model Detail Table")
        model_df = prepared["model_comparison"].copy()
        if selected_models:
            model_df = model_df.loc[model_df["model"].isin(selected_models)]

        model_df = model_df[["model", "accuracy", "f1", "precision", "recall", "auc"]].sort_values("auc", ascending=False)
        st.dataframe(
            model_df.style.format(
                {
                    "accuracy": "{:.4f}",
                    "f1": "{:.4f}",
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "auc": "{:.4f}",
                }
            ),
            use_container_width=True,
            height=340,
        )
        st.markdown('<div class="insight-mini">Etkileşimli tablo, sunum sırasında metrik bazında detaylı inceleme yapılmasını destekler.</div>', unsafe_allow_html=True)


def render_model_governance_dashboard(
    prepared: Dict[str, pd.DataFrame],
    project_root: Path,
    selected_models: List[str],
    theme: Dict[str, str],
) -> None:
    st.markdown('<div class="main-title">Model Governance Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Zorunlu kontroller: tüm modeller için feature importance kapsamı ve MLflow log tamlık denetimi.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Feature Importance Coverage")
    importance_long, coverage_df = build_feature_importance_data(prepared, selected_models)

    if coverage_df.empty:
        st.warning("Model listesi bulunamadı. Feature importance kapsamı doğrulanamıyor.")
    else:
        total_models = int(len(coverage_df))
        logged_models = int(coverage_df["importance_available"].sum())
        coverage_ratio = (logged_models / total_models) if total_models > 0 else 0.0

        fi_col1, fi_col2, fi_col3 = st.columns(3, gap="medium")
        fi_col1.metric("Toplam Model", total_models)
        fi_col2.metric("Importance Loglu Model", logged_models)
        fi_col3.metric("Kapsam Oranı", f"{coverage_ratio:.1%}")

        coverage_view = coverage_df.copy().sort_values(["importance_available", "model"], ascending=[False, True])
        coverage_view["importance_available"] = np.where(coverage_view["importance_available"], "Yes", "No")

        st.dataframe(
            coverage_view.style.format({"top_importance": "{:.4f}"}),
            use_container_width=True,
            height=220,
        )

        missing_importance = coverage_df.loc[~coverage_df["importance_available"], "model"].tolist()
        if missing_importance:
            st.warning(f"Feature importance logu olmayan modeller: {', '.join(missing_importance)}")
        else:
            st.success("Seçili modellerin tamamı için feature importance logu mevcut.")

    if not importance_long.empty:
        global_feature_cap = int(min(30, max(5, importance_long["feature"].nunique())))
        global_top_n = st.slider(
            "Global Top Özellik Sayısı",
            min_value=5,
            max_value=global_feature_cap,
            value=min(12, global_feature_cap),
            step=1,
            key="gov_global_feature_count",
        )

        global_top = (
            importance_long.groupby("feature", as_index=False)["importance_norm"]
            .mean()
            .sort_values("importance_norm", ascending=False)
            .head(global_top_n)
        )

        compare_col1, compare_col2 = st.columns([1.2, 1.0], gap="large")

        with compare_col1:
            st.subheader("Cross-Model Feature Importance Heatmap")
            feature_order = global_top["feature"].tolist()
            heatmap_df = (
                importance_long.loc[importance_long["feature"].isin(feature_order)]
                .pivot_table(index="feature", columns="model", values="importance_norm", aggfunc="mean", fill_value=0.0)
                .reindex(index=list(reversed(feature_order)))
            )

            if not heatmap_df.empty:
                model_order = coverage_df["model"].tolist() if not coverage_df.empty else heatmap_df.columns.tolist()
                model_order = [model for model in model_order if model in heatmap_df.columns]
                heatmap_df = heatmap_df.reindex(columns=model_order)

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_df.values,
                        x=heatmap_df.columns,
                        y=heatmap_df.index,
                        colorscale=[
                            [0.0, hex_to_rgba(theme["accent_blue"], 0.2)],
                            [0.6, theme["accent_blue"]],
                            [1.0, theme["risk_red"]],
                        ],
                        colorbar=dict(title="Normalized"),
                        hovertemplate="Feature=%{y}<br>Model=%{x}<br>Norm Importance=%{z:.4f}<extra></extra>",
                    )
                )
                style_figure(fig_heat, theme, height=430)
                st.plotly_chart(fig_heat, use_container_width=True, config={"displaylogo": False})
                st.markdown(
                    '<div class="insight-mini">Heatmap, hangi özelliklerin birden fazla modelde baskın sinyal ürettiğini gösterir.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Heatmap üretimi için uygun feature importance verisi bulunamadı.")

        with compare_col2:
            st.subheader("Model Bazlı Top Features")
            model_options = sorted(importance_long["model"].dropna().astype(str).unique().tolist())
            selected_importance_model = st.selectbox(
                "Model",
                options=model_options,
                key="gov_feature_model_select",
            )

            model_feature_count = int(
                importance_long.loc[importance_long["model"] == selected_importance_model, "feature"].nunique()
            )
            model_feature_slider_max = max(5, min(40, model_feature_count))
            model_top_n = st.slider(
                "Gösterilecek özellik sayısı",
                min_value=5,
                max_value=model_feature_slider_max,
                value=min(15, model_feature_slider_max),
                step=1,
                key="gov_feature_top_n",
            )

            model_importance = (
                importance_long.loc[importance_long["model"] == selected_importance_model]
                .sort_values("importance", ascending=False)
                .head(model_top_n)
                .sort_values("importance", ascending=True)
            )

            fig_model = px.bar(
                model_importance,
                x="importance",
                y="feature",
                orientation="h",
                color="importance_norm",
                color_continuous_scale=[theme["accent_blue"], theme["risk_red"]],
            )
            fig_model.update_traces(hovertemplate="Feature=%{y}<br>Importance=%{x:.4f}<extra></extra>")
            style_figure(fig_model, theme, height=430)
            st.plotly_chart(fig_model, use_container_width=True, config={"displaylogo": False})
            st.markdown(
                '<div class="insight-mini"><strong>Eksen açıklaması:</strong> X ekseni importance skorunu, Y ekseni özellik adlarını gösterir.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("MLflow Logging Compliance")

    tracking_data = load_mlflow_tracking_data(str(project_root / "mlruns"))
    runs_df = tracking_data["runs"].copy()

    if runs_df.empty:
        st.warning("MLflow run kaydı bulunamadı. mlruns klasörü kontrol edilmelidir.")
        return

    if selected_models:
        selected_set = set(selected_models)
        runs_df = runs_df.loc[
            runs_df["model_name"].isin(selected_set) | runs_df["run_name"].isin(selected_set)
        ]

    if runs_df.empty:
        st.info("Seçili model filtresine uygun MLflow run kaydı yok.")
        return

    filter_col1, filter_col2, filter_col3 = st.columns(3, gap="medium")

    experiment_options = sorted(runs_df["experiment_name"].dropna().astype(str).unique().tolist())
    selected_experiments = filter_col1.multiselect(
        "Experiment Filter",
        options=experiment_options,
        default=experiment_options,
        key="gov_mlflow_experiment_filter",
    )

    status_options = sorted(runs_df["status"].dropna().astype(str).unique().tolist())
    selected_statuses = filter_col2.multiselect(
        "Status Filter",
        options=status_options,
        default=status_options,
        key="gov_mlflow_status_filter",
    )

    only_incomplete = filter_col3.toggle(
        "Sadece eksik logları göster",
        value=False,
        key="gov_mlflow_only_incomplete",
    )

    if selected_experiments:
        runs_df = runs_df.loc[runs_df["experiment_name"].isin(selected_experiments)]
    if selected_statuses:
        runs_df = runs_df.loc[runs_df["status"].isin(selected_statuses)]
    if only_incomplete:
        runs_df = runs_df.loc[~runs_df["fully_logged"]]

    if runs_df.empty:
        st.info("Seçilen filtrelerle gösterilecek MLflow run bulunamadı.")
        return

    total_runs = int(len(runs_df))
    fully_logged_runs = int(runs_df["fully_logged"].sum())
    compliance_ratio = (fully_logged_runs / total_runs) if total_runs > 0 else 0.0
    unique_experiments = int(runs_df["experiment_id"].nunique())

    ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4, gap="medium")
    ml_col1.metric("Toplam Run", total_runs)
    ml_col2.metric("Tam Loglanan Run", fully_logged_runs)
    ml_col3.metric("Uygunluk", f"{compliance_ratio:.1%}")
    ml_col4.metric("Experiment Sayısı", unique_experiments)

    compliance_by_model = (
        runs_df.groupby("model_name", as_index=False)
        .agg(run_count=("run_id", "count"), compliance_rate=("fully_logged", "mean"))
        .sort_values("compliance_rate", ascending=False)
    )
    compliance_by_model["compliance_pct"] = compliance_by_model["compliance_rate"] * 100.0

    fig_compliance = px.bar(
        compliance_by_model,
        x="model_name",
        y="compliance_pct",
        color="compliance_pct",
        color_continuous_scale=[theme["risk_red"], theme["warning_orange"], theme["success_green"]],
        hover_data={"run_count": True},
    )
    fig_compliance.update_traces(
        hovertemplate="Model=%{x}<br>Compliance=%{y:.1f}%<br>Run Count=%{customdata[0]}<extra></extra>"
    )
    fig_compliance.update_yaxes(title_text="Compliance %", range=[0, 100])
    style_figure(fig_compliance, theme, height=360)
    st.plotly_chart(fig_compliance, use_container_width=True, config={"displaylogo": False})

    runs_view = runs_df.copy().sort_values("start_time", ascending=False)
    runs_view["start_time"] = pd.to_datetime(runs_view["start_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    runs_view["start_time"] = runs_view["start_time"].fillna("N/A")
    runs_view["fully_logged"] = np.where(runs_view["fully_logged"], "Yes", "No")

    display_columns = [
        "experiment_name",
        "model_name",
        "run_name",
        "status",
        "fully_logged",
        "has_params",
        "has_metrics",
        "has_model_artifact",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "start_time",
        "run_id",
    ]
    st.dataframe(
        runs_view[display_columns].style.format(
            {
                "accuracy": "{:.4f}",
                "f1": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "auc": "{:.4f}",
            }
        ),
        use_container_width=True,
        height=330,
    )

    run_lookup = runs_df.set_index("run_id")[["model_name", "status"]].to_dict("index")
    run_ids = runs_df["run_id"].dropna().astype(str).tolist()

    selected_run_id = st.selectbox(
        "Run Detay İncelemesi",
        options=run_ids,
        format_func=lambda rid: f"{rid[:8]} | {run_lookup[rid]['model_name']} | {run_lookup[rid]['status']}",
        key="gov_mlflow_run_detail_select",
    )

    run_row = runs_df.loc[runs_df["run_id"] == selected_run_id].iloc[0]
    missing_items = []
    if not bool(run_row["has_params"]):
        missing_items.append("parameters")
    if not bool(run_row["has_metrics"]):
        missing_items.append("metrics")
    if not bool(run_row["has_model_artifact"]):
        missing_items.append("model")

    if missing_items:
        st.error(f"Seçili run için eksik zorunlu log alanları: {', '.join(missing_items)}")
    else:
        st.success("Seçili run, zorunlu MLflow log alanlarının tamamını içeriyor.")

    detail_col1, detail_col2 = st.columns(2, gap="large")

    with detail_col1:
        st.markdown("**Run Parameters**")
        params_df = tracking_data["params"].copy()
        run_params = params_df.loc[params_df["run_id"] == selected_run_id, ["key", "value"]]
        if run_params.empty:
            st.info("Parameter logu bulunamadı.")
        else:
            st.dataframe(run_params.sort_values("key"), use_container_width=True, height=260)

    with detail_col2:
        st.markdown("**Run Metrics**")
        metrics_df = tracking_data["metrics"].copy()
        run_metrics = metrics_df.loc[metrics_df["run_id"] == selected_run_id, ["metric", "value", "step", "timestamp"]]
        if run_metrics.empty:
            st.info("Metric logu bulunamadı.")
        else:
            run_metrics = run_metrics.sort_values("metric")
            run_metrics["timestamp"] = pd.to_datetime(run_metrics["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            run_metrics["timestamp"] = run_metrics["timestamp"].fillna("N/A")
            st.dataframe(
                run_metrics.style.format({"value": "{:.6f}"}),
                use_container_width=True,
                height=260,
            )

    artifact_df = tracking_data["model_artifacts"].copy()
    run_artifact = artifact_df.loc[
        artifact_df["run_id"] == selected_run_id,
        ["artifact_path", "flavors", "model_class", "created_utc", "model_size_bytes"],
    ]
    st.markdown("**Model Artifact**")
    if run_artifact.empty:
        st.info("Model artifact logu bulunamadı.")
    else:
        st.dataframe(
            run_artifact.style.format({"model_size_bytes": "{:.0f}"}),
            use_container_width=True,
            height=120,
        )

    with st.expander("Run Tags"):
        tags_df = tracking_data["tags"].copy()
        run_tags = tags_df.loc[tags_df["run_id"] == selected_run_id, ["key", "value"]]
        if run_tags.empty:
            st.info("Tag kaydı bulunamadı.")
        else:
            st.dataframe(run_tags.sort_values("key"), use_container_width=True, height=220)


def render_threshold_dashboard(
    prepared: Dict[str, pd.DataFrame],
    kpi_map: Dict[str, object],
    selected_models: List[str],
    selected_threshold: float,
    theme: Dict[str, str],
) -> None:
    st.markdown('<div class="main-title">Threshold & Business Strategy Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">F1 skoru, maliyet duyarlılığı ve hata etkisi analizi ile eşik politikasını optimize edin.</div>',
        unsafe_allow_html=True,
    )

    threshold_df = prepared["threshold_tuning_best_model"].copy()
    if threshold_df.empty:
        st.warning("Threshold dataset is empty.")
        return

    if selected_models:
        filtered = threshold_df.loc[threshold_df["model"].isin(selected_models)]
        if not filtered.empty:
            threshold_df = filtered

    available_models = sorted(threshold_df["model"].dropna().unique().tolist())
    default_model = "GBTClassifier" if "GBTClassifier" in available_models else available_models[0]

    strategy_col1, strategy_col2, strategy_col3 = st.columns(3, gap="medium")

    rec_cost = format_value(kpi_map.get("Recommended Threshold (Cost)", np.nan), "threshold")
    rec_f1 = format_value(kpi_map.get("Recommended Threshold (F1)", np.nan), "threshold")

    with strategy_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Recommended Threshold</div>
                <div class="kpi-value">Cost: {rec_cost}</div>
                <div class="kpi-value" style="font-size:1.1rem; margin-top:0.25rem;">F1: {rec_f1}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with strategy_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Selected Threshold</div>
                <div class="kpi-value">{selected_threshold:.2f}</div>
                <div class="insight-mini" style="margin-bottom:0;">Politika duyarlılığını test etmek için kenar çubuğundaki sliderı kullanın.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with strategy_col3:
        st.markdown(
            """
            <div class="insight-card">
                Kredi risk tahmininde yanlış negatiflerin maliyeti belirgin şekilde daha yüksektir.
                <br><br>
                FN maliyetinin baskın olduğu senaryolarda recall odaklı eşik noktaları önceliklendirilmelidir.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    cost_scenarios = (
        threshold_df[["fp_cost", "fn_cost"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["fp_cost", "fn_cost"])
    )
    if cost_scenarios.empty:
        st.warning("No cost scenarios found in threshold data.")
        return

    scenario_labels = [f"FP={int(row.fp_cost)}, FN={int(row.fn_cost)}" for row in cost_scenarios.itertuples(index=False)]

    if "fp_cost_base" in kpi_map and "fn_cost_base" in kpi_map:
        base_label = f"FP={int(float(kpi_map['fp_cost_base']))}, FN={int(float(kpi_map['fn_cost_base']))}"
        default_scenario_index = scenario_labels.index(base_label) if base_label in scenario_labels else 0
    else:
        default_scenario_index = 0

    selected_model = st.selectbox("Strategy Model", options=available_models, index=available_models.index(default_model))
    selected_scenario = st.selectbox("Cost Scenario", options=scenario_labels, index=default_scenario_index)

    fp_sel = float(selected_scenario.split(",")[0].split("=")[1])
    fn_sel = float(selected_scenario.split(",")[1].split("=")[1])

    strategy_df = threshold_df.loc[
        (threshold_df["model"] == selected_model)
        & (threshold_df["fp_cost"] == fp_sel)
        & (threshold_df["fn_cost"] == fn_sel)
    ].sort_values("threshold")

    if strategy_df.empty:
        st.warning("No threshold records available for selected model/scenario.")
        return

    nearest_idx = (strategy_df["threshold"] - selected_threshold).abs().idxmin()
    nearest_row = strategy_df.loc[nearest_idx]

    line_col1, line_col2 = st.columns(2, gap="large")

    with line_col1:
        st.subheader("Threshold vs F1")
        fig_f1 = go.Figure()
        fig_f1.add_trace(
            go.Scatter(
                x=strategy_df["threshold"],
                y=strategy_df["f1"],
                mode="lines+markers",
                line=dict(color=theme["accent_blue"], width=3),
                marker=dict(size=6),
                name="F1",
                hovertemplate="Threshold=%{x:.2f}<br>F1=%{y:.4f}<extra></extra>",
            )
        )
        fig_f1.add_vline(x=selected_threshold, line_width=2, line_dash="dash", line_color=theme["warning_orange"])
        style_figure(fig_f1, theme, height=370)
        st.plotly_chart(fig_f1, use_container_width=True, config={"displaylogo": False})
        st.markdown(
            f'<div class="insight-mini">Seçili maliyet senaryosunda {selected_threshold:.2f} eşiğinde F1 skoru {nearest_row["f1"]:.4f} olarak hesaplandı.</div>',
            unsafe_allow_html=True,
        )

    with line_col2:
        st.subheader("Threshold vs Expected Cost")
        fig_cost = go.Figure()
        fig_cost.add_trace(
            go.Scatter(
                x=strategy_df["threshold"],
                y=strategy_df["expected_cost"],
                mode="lines+markers",
                line=dict(color=theme["risk_red"], width=3),
                marker=dict(size=6),
                name="Expected Cost",
                hovertemplate="Threshold=%{x:.2f}<br>Cost=%{y:,.0f}<extra></extra>",
            )
        )
        fig_cost.add_vline(x=selected_threshold, line_width=2, line_dash="dash", line_color=theme["warning_orange"])
        style_figure(fig_cost, theme, height=370)
        st.plotly_chart(fig_cost, use_container_width=True, config={"displaylogo": False})
        st.markdown(
            f'<div class="insight-mini">{selected_threshold:.2f} eşiğinde beklenen maliyet {nearest_row["expected_cost"]:,.0f} düzeyindedir.</div>',
            unsafe_allow_html=True,
        )

    row3_col1, row3_col2 = st.columns(2, gap="large")

    with row3_col1:
        st.subheader("FP/FN Cost Impact")
        cost_matrix = prepared["business_cost_matrix"].copy()
        cost_matrix = cost_matrix.loc[cost_matrix["model"] == selected_model].copy()
        cost_matrix["scenario"] = cost_matrix.apply(
            lambda r: f"FP={int(r['fp_cost'])}, FN={int(r['fn_cost'])}", axis=1
        )
        impact_fig = px.bar(
            cost_matrix.sort_values("minimum_expected_cost", ascending=False),
            x="scenario",
            y="minimum_expected_cost",
            color="minimum_expected_cost",
            color_continuous_scale=[theme["accent_blue"], theme["risk_red"]],
            hover_data={
                "optimal_threshold": ":.2f",
                "f1_at_optimal": ":.4f",
                "recall_at_optimal": ":.4f",
                "precision_at_optimal": ":.4f",
            },
        )
        impact_fig.update_traces(
            hovertemplate=(
                "Scenario=%{x}<br>Min Cost=%{y:,.0f}<br>"
                "Optimal Threshold=%{customdata[0]:.2f}<br>"
                "F1=%{customdata[1]:.4f}<extra></extra>"
            )
        )
        style_figure(impact_fig, theme, height=370)
        st.plotly_chart(impact_fig, use_container_width=True, config={"displaylogo": False})
        st.markdown('<div class="insight-mini">FN maliyeti arttıkça optimal eşik, recall lehine bölgelere kayma eğilimi gösterir.</div>', unsafe_allow_html=True)

    with row3_col2:
        st.subheader("Business Cost Matrix Heatmap")
        heatmap_df = prepared["business_cost_matrix"].copy()
        heatmap_df = heatmap_df.loc[heatmap_df["model"] == selected_model]
        heat_pivot = heatmap_df.pivot_table(
            index="fn_cost", columns="fp_cost", values="minimum_expected_cost", aggfunc="min"
        )

        heat_fig = go.Figure(
            data=go.Heatmap(
                z=heat_pivot.values,
                x=[f"FP={int(v)}" for v in heat_pivot.columns],
                y=[f"FN={int(v)}" for v in heat_pivot.index],
                colorscale=[
                    [0.0, hex_to_rgba(theme["accent_blue"], 0.25)],
                    [0.5, theme["warning_orange"]],
                    [1.0, theme["risk_red"]],
                ],
                hovertemplate="%{y} | %{x}<br>Min Cost=%{z:,.0f}<extra></extra>",
            )
        )
        style_figure(heat_fig, theme, height=370)
        st.plotly_chart(heat_fig, use_container_width=True, config={"displaylogo": False})
        st.markdown('<div class="insight-mini">Heatmap, farklı FP/FN varsayımlarında politika sonuçlarının hızlı karşılaştırılmasını sağlar.</div>', unsafe_allow_html=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent

    data_dir = find_dashboard_data_dir(project_root)
    if data_dir is None:
        st.error("Dashboard data directory not found. Expected output/step7/dashboard_pack or output/step6.")
        st.stop()

    missing_files = get_missing_files(data_dir)
    if missing_files:
        st.error("Missing required files:")
        for filename in missing_files:
            st.write(f"- {filename}")
        st.stop()

    raw_dataset_path = find_raw_dataset_path(project_root)

    with st.spinner("Loading datasets and preparing interactive visuals..."):
        loaded = load_dashboard_data(str(data_dir))
        prepared = prepare_data(loaded)
        raw_eda = pd.DataFrame()
        if raw_dataset_path is not None:
            raw_eda = prepare_raw_eda(load_raw_eda_data(str(raw_dataset_path)))

    model_options = sorted(prepared["model_comparison"]["model"].dropna().unique().tolist())

    monthly_df = prepared["monthly_default_trend"].dropna(subset=["issue_month"]).sort_values("issue_month")
    min_date = monthly_df["issue_month"].min().date() if not monthly_df.empty else None
    max_date = monthly_df["issue_month"].max().date() if not monthly_df.empty else None

    threshold_df = prepared["threshold_tuning_best_model"].dropna(subset=["threshold"])
    threshold_min = float(threshold_df["threshold"].min()) if not threshold_df.empty else 0.0
    threshold_max = float(threshold_df["threshold"].max()) if not threshold_df.empty else 1.0

    kpi_map = get_kpi_map(prepared["kpi_cards"])
    default_threshold = kpi_map.get("Recommended Threshold (Cost)", 0.4)
    try:
        default_threshold = float(default_threshold)
    except (TypeError, ValueError):
        default_threshold = 0.4
    default_threshold = max(threshold_min, min(threshold_max, default_threshold))

    st.sidebar.markdown("## Dashboard Controls")

    dark_mode = st.sidebar.toggle("Dark Mode", value=True)
    theme = THEMES["dark" if dark_mode else "light"]
    apply_theme_css(theme)

    page = st.sidebar.radio(
        "Page",
        [
            "Executive Overview",
            "EDA Dashboard",
            "Machine Learning Dashboard",
            "Model Governance Dashboard",
            "Threshold & Business Strategy Dashboard",
        ],
    )

    selected_models = st.sidebar.multiselect(
        "Model Filter",
        options=model_options,
        default=model_options,
    )

    if min_date and max_date:
        date_selection = st.sidebar.date_input(
            "Date Filter",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_selection, tuple) and len(date_selection) == 2:
            selected_start, selected_end = date_selection
        else:
            selected_start, selected_end = date_selection, date_selection

        monthly_filtered = monthly_df.loc[
            (monthly_df["issue_month"].dt.date >= selected_start)
            & (monthly_df["issue_month"].dt.date <= selected_end)
        ]
    else:
        monthly_filtered = monthly_df

    selected_threshold = st.sidebar.slider(
        "Threshold Slider",
        min_value=round(threshold_min, 2),
        max_value=round(threshold_max, 2),
        value=round(default_threshold, 2),
        step=0.01,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data Source: {data_dir.as_posix()}")
    if raw_dataset_path:
        st.sidebar.caption("Raw EDA Source: available")
    else:
        st.sidebar.caption("Raw EDA Source: not found")

    if page == "Executive Overview":
        render_executive_overview(prepared, kpi_map, monthly_filtered, theme)
    elif page == "EDA Dashboard":
        render_eda_dashboard(prepared, raw_eda, monthly_filtered, theme)
    elif page == "Machine Learning Dashboard":
        render_ml_dashboard(prepared, selected_models, kpi_map, theme)
    elif page == "Model Governance Dashboard":
        render_model_governance_dashboard(prepared, project_root, selected_models, theme)
    else:
        render_threshold_dashboard(prepared, kpi_map, selected_models, selected_threshold, theme)


if __name__ == "__main__":
    main()
