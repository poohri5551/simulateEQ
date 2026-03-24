from pathlib import Path
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from app.features import clean_training_frame, make_features


def load_table_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(path, encoding="latin1")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    df["source_file"] = path.name
    return df


def load_input_folder_or_file(input_path: str) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        df = load_table_file(path)
        print(f"Loaded: {path.name} ({len(df)} rows)")
        return df

    files = []
    files.extend(sorted(path.glob("*.xlsx")))
    files.extend(sorted(path.glob("*.xls")))
    files.extend(sorted(path.glob("*.csv")))

    if not files:
        raise FileNotFoundError(f"No .xlsx/.xls/.csv files found in folder: {path}")

    frames = []
    for f in files:
        try:
            tmp = load_table_file(f)
            frames.append(tmp)
            print(f"Loaded: {f.name} ({len(tmp)} rows)")
        except Exception as e:
            print(f"Skip file: {f.name} -> {e}")

    if not frames:
        raise ValueError(f"Could not read any supported files from folder: {path}")

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def choose_cv(clean_df: pd.DataFrame):
    """
    ใช้ GroupKFold ถ้ามี event_id หลาย event พอ
    ไม่งั้น fallback เป็น KFold
    """
    if "event_id" in clean_df.columns:
        groups = clean_df["event_id"].astype(str)
        n_groups = groups.nunique()
        if n_groups >= 5:
            return GroupKFold(n_splits=5), groups, f"GroupKFold(5) by event_id [n_groups={n_groups}]"
        elif n_groups >= 3:
            return GroupKFold(n_splits=3), groups, f"GroupKFold(3) by event_id [n_groups={n_groups}]"

    return KFold(n_splits=5, shuffle=True, random_state=42), None, "KFold(5, shuffle=True, random_state=42)"


def build_candidates():
    return {
        "rf_reg": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "et_reg": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "hgb_reg": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=350,
            min_samples_leaf=10,
            l2_regularization=0.0,
            random_state=42,
        ),
    }


def evaluate_model(model, X, y, cv, groups=None):
    if groups is not None:
        y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)
    else:
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5

    y_pred_round = np.clip(np.rint(y_pred), 1, 12)
    y_true_round = np.clip(np.rint(y), 1, 12)

    acc_pm1 = float(np.mean(np.abs(y_pred_round - y_true_round) <= 1))
    acc_exact = float(np.mean(y_pred_round == y_true_round))

    return {
        "cv_mae": float(mae),
        "cv_rmse": float(rmse),
        "cv_accuracy_pm1": float(acc_pm1),
        "cv_accuracy_exact": float(acc_exact),
        "y_pred_min": float(np.min(y_pred)),
        "y_pred_max": float(np.max(y_pred)),
    }


def fit_best_model(df: pd.DataFrame):
    clean_df = clean_training_frame(df)

    X_raw = clean_df.drop(columns=["event_id", "mmi"], errors="ignore")
    y = clean_df["mmi"].astype(float)
    X = make_features(X_raw)

    cv, groups, cv_name = choose_cv(clean_df)
    candidates = build_candidates()
    all_results = {}

    best_name = None
    best_score = float("inf")   # ใช้ MAE ต่ำสุดเป็นตัวเลือกหลัก
    best_model = None

    print("\nUsing CV:", cv_name)

    for name, model in candidates.items():
        print(f"\n=== Evaluating {name} ===")
        result = evaluate_model(model, X, y, cv=cv, groups=groups)
        result["cv_strategy"] = cv_name
        all_results[name] = result

        print("CV MAE:", round(result["cv_mae"], 4))
        print("CV RMSE:", round(result["cv_rmse"], 4))
        print("CV Accuracy (±1 level):", round(result["cv_accuracy_pm1"], 4))
        print("CV Accuracy (exact):", round(result["cv_accuracy_exact"], 4))
        print("Pred min/max:", round(result["y_pred_min"], 4), "/", round(result["y_pred_max"], 4))

        if result["cv_mae"] < best_score:
            best_score = result["cv_mae"]
            best_name = name
            best_model = clone(model)

    best_model.fit(X, y)

    report = {
        "n_rows": int(len(clean_df)),
        "raw_columns": list(df.columns),
        "feature_columns_raw": list(X_raw.columns),
        "feature_columns_engineered": list(X.columns),
        "best_model_name": best_name,
        "selection_metric": "cv_mae",
        "best_cv_mae": float(best_score),
        "cv_strategy": cv_name,
        "all_results": all_results,
        "note": (
            "Regression training for MMI. "
            "Use cv_mae as main selection metric; "
            "check cv_accuracy_exact and cv_accuracy_pm1 as secondary metrics."
        ),
    }

    return best_model, report, clean_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to file OR folder containing .xlsx/.xls/.csv")
    ap.add_argument("--out-model", default="app/models/mmi_model.joblib")
    ap.add_argument("--out-report", default="reports/train_report_regression.json")
    args = ap.parse_args()

    df = load_input_folder_or_file(args.excel)
    model, report, clean_df = fit_best_model(df)

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.out_model)
    Path(args.out_report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== BEST MODEL ===")
    print("Rows used:", len(clean_df))
    print("Best model:", report["best_model_name"])
    print("Selection metric:", report["selection_metric"])
    print("Best CV MAE:", round(report["best_cv_mae"], 4))

    best_result = report["all_results"][report["best_model_name"]]
    print("Best CV RMSE:", round(best_result["cv_rmse"], 4))
    print("Best CV Accuracy (±1 level):", round(best_result["cv_accuracy_pm1"], 4))
    print("Best CV Accuracy (exact):", round(best_result["cv_accuracy_exact"], 4))

    print("Saved model ->", args.out_model)
    print("Saved report ->", args.out_report)


if __name__ == "__main__":
    main()