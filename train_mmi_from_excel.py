from pathlib import Path
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.features import clean_training_frame, make_features, build_model


def load_excel_input(excel_input: str) -> pd.DataFrame:
    path = Path(excel_input)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # กรณีเป็นไฟล์ Excel ไฟล์เดียว
    if path.is_file():
        if path.suffix.lower() not in [".xlsx", ".xls"]:
            raise ValueError(f"Not an Excel file: {path}")
        df = pd.read_excel(path)
        df["source_file"] = path.name
        return df

    # กรณีเป็นโฟลเดอร์: อ่านทุกไฟล์ Excel แล้วรวมกัน
    excel_files = sorted(list(path.glob("*.xlsx")) + list(path.glob("*.xls")))

    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in folder: {path}")

    frames = []
    for f in excel_files:
        try:
            tmp = pd.read_excel(f)
            tmp["source_file"] = f.name
            frames.append(tmp)
            print(f"Loaded: {f.name} ({len(tmp)} rows)")
        except Exception as e:
            print(f"Skip file: {f.name} -> {e}")

    if not frames:
        raise ValueError(f"Could not read any Excel files from folder: {path}")

    df = pd.concat(frames, ignore_index=True)
    return df


def fit_and_report(df: pd.DataFrame, model_name: str):
    clean_df = clean_training_frame(df)

    X_raw = clean_df.drop(columns=["event_id", "mmi"])
    y = clean_df["mmi"].astype(float)

    X = make_features(X_raw)
    model = build_model(model_name)

    loo = LeaveOneOut()
    y_pred_cv = cross_val_predict(model, X, y, cv=loo)

    mae = mean_absolute_error(y, y_pred_cv)
    rmse = mean_squared_error(y, y_pred_cv) ** 0.5
    acc_pm1 = float(np.mean(np.abs(np.rint(y_pred_cv) - y) <= 1))
    acc_exact = float(np.mean(np.rint(y_pred_cv) == y))

    model.fit(X, y)

    report = {
        "n_rows": int(len(clean_df)),
        "raw_columns": list(df.columns),
        "feature_columns_raw": list(X_raw.columns),
        "feature_columns_engineered": list(X.columns),
        "model_name": model_name,
        "cv_strategy": "LeaveOneOut",
        "cv_mae": float(mae),
        "cv_rmse": float(rmse),
        "cv_accuracy_pm1": acc_pm1,
        "cv_accuracy_exact": acc_exact,
        "note": "Dataset is very small, so metrics are unstable and only for rough inspection.",
    }
    return model, report, clean_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--excel",
        required=True,
        help="Path to Excel file OR folder containing Excel files"
    )
    ap.add_argument("--model-name", default="rf", choices=["rf", "hgb"])
    ap.add_argument("--out-model", default="app/models/mmi_model.joblib")
    ap.add_argument("--out-report", default="reports/train_report.json")
    args = ap.parse_args()

    df = load_excel_input(args.excel)

    model, report, clean_df = fit_and_report(df, args.model_name)

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.out_model)
    Path(args.out_report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Rows used:", len(clean_df))
    print("Raw feature columns:", report["feature_columns_raw"])
    print("Engineered feature columns:", report["feature_columns_engineered"])
    print("CV MAE:", round(report["cv_mae"], 4))
    print("CV RMSE:", round(report["cv_rmse"], 4))
    print("CV Accuracy (±1 level):", round(report["cv_accuracy_pm1"], 4))
    print("CV Accuracy (exact):", round(report["cv_accuracy_exact"], 4))
    print("Saved model ->", args.out_model)
    print("Saved report ->", args.out_report)


if __name__ == "__main__":
    main()