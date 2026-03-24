from pathlib import Path
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

from app.features import clean_training_frame, make_features


def load_table_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        # รองรับ UTF-8 และกัน encoding พังเบื้องต้น
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

    # กันคอลัมน์ซ้ำหลัง concat/normalize
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df


def build_candidates():
    return {
        "rf_cls": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "et_cls": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "hgb_cls": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            min_samples_leaf=10,
            random_state=42,
        ),
    }


def evaluate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5
    acc_exact = accuracy_score(y, y_pred)
    acc_pm1 = float(np.mean(np.abs(y_pred - y) <= 1))

    return {
        "cv_strategy": "StratifiedKFold(5)",
        "cv_mae": float(mae),
        "cv_rmse": float(rmse),
        "cv_accuracy_pm1": float(acc_pm1),
        "cv_accuracy_exact": float(acc_exact),
    }


def fit_best_model(df: pd.DataFrame):
    clean_df = clean_training_frame(df)

    X_raw = clean_df.drop(columns=["event_id", "mmi"])
    y = clean_df["mmi"].astype(float).round().astype(int)

    X = make_features(X_raw)

    candidates = build_candidates()
    all_results = {}

    best_name = None
    best_score = -1.0
    best_model = None

    for name, model in candidates.items():
        print(f"\n=== Evaluating {name} ===")
        result = evaluate_model(model, X, y)
        all_results[name] = result

        print("CV MAE:", round(result["cv_mae"], 4))
        print("CV RMSE:", round(result["cv_rmse"], 4))
        print("CV Accuracy (±1 level):", round(result["cv_accuracy_pm1"], 4))
        print("CV Accuracy (exact):", round(result["cv_accuracy_exact"], 4))

        if result["cv_accuracy_exact"] > best_score:
            best_score = result["cv_accuracy_exact"]
            best_name = name
            best_model = model

    best_model.fit(X, y)

    report = {
        "n_rows": int(len(clean_df)),
        "raw_columns": list(df.columns),
        "feature_columns_raw": list(X_raw.columns),
        "feature_columns_engineered": list(X.columns),
        "best_model_name": best_name,
        "best_cv_accuracy_exact": float(best_score),
        "all_results": all_results,
        "note": "Classifier-based training for improving exact MMI accuracy.",
    }

    return best_model, report, clean_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to file OR folder containing .xlsx/.xls/.csv")
    ap.add_argument("--out-model", default="app/models/mmi_model.joblib")
    ap.add_argument("--out-report", default="reports/train_report_classifier.json")
    args = ap.parse_args()

    df = load_input_folder_or_file(args.excel)
    model, report, clean_df = fit_best_model(df)

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.out_model)
    Path(args.out_report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== BEST MODEL ===")
    print("Rows used:", len(clean_df))
    print("Best model:", report["best_model_name"])
    print("Best CV Accuracy (exact):", round(report["best_cv_accuracy_exact"], 4))
    print("Saved model ->", args.out_model)
    print("Saved report ->", args.out_report)


if __name__ == "__main__":
    main()