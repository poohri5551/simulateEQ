import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

BASE_FEATURES = ["dist", "pga"]
OPTIONAL_FEATURES = ["mag", "lat", "lon"]

ROMAN_TO_INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}


def _dedupe_columns_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    # ถ้าชื่อคอลัมน์ซ้ำหลัง normalize ให้เก็บตัวแรก
    return df.loc[:, ~df.columns.duplicated()].copy()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    rename_map = {
        "distance": "dist",
        "distance_km": "dist",
        "distance(km)": "dist",
        "pga_g": "pga",
        "mmi_level": "mmi",
        "mmi(ai)": "mmi",
        "mmi ": "mmi",
    }
    out = out.rename(columns=rename_map)

    # สำคัญมาก: หลัง strip/rename อาจมีชื่อซ้ำ เช่น lat กับ lat 
    out = _dedupe_columns_keep_first(out)
    return out


def convert_mmi_to_numeric(s):
    if pd.isna(s):
        return np.nan

    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)

    txt = str(s).strip().upper()
    if txt in ROMAN_TO_INT:
        return float(ROMAN_TO_INT[txt])

    try:
        return float(txt)
    except Exception:
        return np.nan


def get_feature_columns(df: pd.DataFrame):
    cols = []
    for c in BASE_FEATURES + OPTIONAL_FEATURES:
        if c in df.columns:
            cols.append(c)
    return cols


def validate_columns(df: pd.DataFrame):
    required = ["mmi", "dist", "pga"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    validate_columns(df)

    if "event_id" not in df.columns:
        df["event_id"] = np.arange(1, len(df) + 1)

    df["mmi"] = df["mmi"].apply(convert_mmi_to_numeric)

    use_cols = ["event_id", "mmi"] + get_feature_columns(df)
    out = df[use_cols].copy()

    # กันคอลัมน์ซ้ำอีกรอบ เผื่อมาจากหลายไฟล์
    out = _dedupe_columns_keep_first(out)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["mmi", "dist", "pga"]).reset_index(drop=True)

    out["dist"] = pd.to_numeric(out["dist"], errors="coerce")
    out["pga"] = pd.to_numeric(out["pga"], errors="coerce")
    out["mmi"] = pd.to_numeric(out["mmi"], errors="coerce")

    if "mag" in out.columns:
        out["mag"] = pd.to_numeric(out["mag"], errors="coerce")
    if "lat" in out.columns:
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    if "lon" in out.columns:
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["mmi", "dist", "pga"])

    out = out[out["dist"] > 0].copy()
    out = out[out["pga"] > 0].copy()

    return out.reset_index(drop=True)


def make_features(X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X = normalize_columns(X)
    X = _dedupe_columns_keep_first(X)

    out = pd.DataFrame(index=X.index)

    dist = pd.to_numeric(X["dist"], errors="coerce").astype(float)
    pga = pd.to_numeric(X["pga"], errors="coerce").astype(float)

    has_mag = "mag" in X.columns
    has_lat = "lat" in X.columns
    has_lon = "lon" in X.columns

    if has_mag:
        mag = pd.to_numeric(X["mag"], errors="coerce").astype(float)
    if has_lat:
        lat = pd.to_numeric(X["lat"], errors="coerce").astype(float)
    if has_lon:
        lon = pd.to_numeric(X["lon"], errors="coerce").astype(float)

    eps = 1e-6
    log_dist = np.log10(np.clip(dist, eps, None))
    log_pga = np.log10(np.clip(pga, eps, None))

    out["dist"] = dist
    out["pga"] = pga
    out["log_dist"] = log_dist
    out["log_pga"] = log_pga
    out["inv_dist"] = 1.0 / np.clip(dist, 1.0, None)
    out["dist_sq"] = dist ** 2
    out["log_dist_sq"] = log_dist ** 2
    out["pga_over_dist"] = pga / np.clip(dist, 1.0, None)

    if has_mag:
        out["mag"] = mag
        out["mag_x_log_dist"] = mag * log_dist
        out["mag_x_log_pga"] = mag * log_pga
        out["mag_over_log_dist1p"] = mag / (np.abs(log_dist) + 1.0)

    if has_lat:
        out["lat"] = lat
    if has_lon:
        out["lon"] = lon
    if has_lat and has_lon:
        out["lat_lon"] = lat * lon

    return out


def build_model(model_name: str):
    name = model_name.lower()

    if name in {"hgb", "histgb", "hist_gradient_boosting"}:
        return HistGradientBoostingRegressor(
            random_state=42,
            max_depth=4,
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=2,
            l2_regularization=0.0,
        )

    if name in {"rf", "random_forest", "randomforest"}:
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")