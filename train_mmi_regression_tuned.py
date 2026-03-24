from pathlib import Path
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)

from app.features import clean_training_frame, make_features


warnings.filterwarnings("ignore", category=UserWarning)


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


def load_input_folder_or_file(input_path: str, include_csv: bool = False) -> pd.DataFrame:
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

    if include_csv:
        files.extend(sorted(path.glob("*.csv")))

    if not files:
        raise FileNotFoundError(f"No supported files found in folder: {path}")

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


def prepare_dataset(df: pd.DataFrame):
    clean_df = clean_training_frame(df)

    if "source_file" in clean_df.columns and "event_id" not in clean_df.columns:
        clean_df["event_id"] = clean_df["source_file"].astype(str)

    X_raw = clean_df.drop(columns=["event_id", "mmi"], errors="ignore")
    y = clean_df["mmi"].astype(float)
    X = make_features(X_raw)

    groups = clean_df["event_id"].astype(str) if "event_id" in clean_df.columns else None
    return clean_df, X_raw, X, y, groups


def choose_cv(groups):
    if groups is not None and pd.Series(groups).nunique() >= 5:
        return GroupKFold(n_splits=5), "GroupKFold(5) by event_id"
    return KFold(n_splits=5, shuffle=True, random_state=42), "KFold(5, shuffle=True, random_state=42)"


def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    y_pred_round = np.clip(np.rint(y_pred), 1, 12)
    y_true_round = np.clip(np.rint(y_true), 1, 12)

    acc_pm1 = float(np.mean(np.abs(y_pred_round - y_true_round) <= 1))
    acc_exact = float(np.mean(y_pred_round == y_true_round))

    return {
        "cv_mae": float(mae),
        "cv_rmse": float(rmse),
        "cv_accuracy_pm1": float(acc_pm1),
        "cv_accuracy_exact": float(acc_exact),
    }


def build_search_spaces():
    return {
        "rf_reg": (
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
            ),
            {
                "n_estimators": [300, 500, 700, 900],
                "max_depth": [None, 8, 12, 16, 24],
                "min_samples_leaf": [1, 2, 4, 8],
                "min_samples_split": [2, 5, 10, 20],
                "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
            },
        ),
        "et_reg": (
            ExtraTreesRegressor(
                random_state=42,
                n_jobs=-1,
            ),
            {
                "n_estimators": [300, 500, 700, 900],
                "max_depth": [None, 8, 12, 16, 24],
                "min_samples_leaf": [1, 2, 4, 8],
                "min_samples_split": [2, 5, 10, 20],
                "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
            },
        ),
        "hgb_reg": (
            HistGradientBoostingRegressor(
                random_state=42,
            ),
            {
                "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
                "max_depth": [None, 4, 6, 8, 10],
                "max_iter": [200, 300, 500, 700],
                "min_samples_leaf": [5, 10, 20, 30],
                "l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        ),
    }


def tune_one_model(name, model, param_dist, X, y, cv, groups=None, n_iter=20):
    print(f"\n=== Tuning {name} ===")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    if groups is not None and isinstance(cv, GroupKFold):
        search.fit(X, y, groups=groups)
        y_pred = cross_val_predict(search.best_estimator_, X, y, cv=cv, groups=groups, n_jobs=-1)
    else:
        search.fit(X, y)
        y_pred = cross_val_predict(search.best_estimator_, X, y, cv=cv, n_jobs=-1)

    result = evaluate_predictions(y, y_pred)
    result["best_params"] = search.best_params_
    result["best_search_score"] = float(search.best_score_)

    print("Best params:", search.best_params_)
    print("CV MAE:", round(result["cv_mae"], 4))
    print("CV RMSE:", round(result["cv_rmse"], 4))
    print("CV Accuracy (±1 level):", round(result["cv_accuracy_pm1"], 4))
    print("CV Accuracy (exact):", round(result["cv_accuracy_exact"], 4))

    return search.best_estimator_, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to file or folder")
    ap.add_argument("--include-csv", action="store_true", help="Include CSV files when input is a folder")
    ap.add_argument("--n-iter", type=int, default=20, help="RandomizedSearch iterations per model")
    ap.add_argument("--out-model", default="app/models/mmi_model.joblib")
    ap.add_argument("--out-report", default="reports/train_report_regression_tuned.json")
    args = ap.parse_args()

    df = load_input_folder_or_file(args.excel, include_csv=args.include_csv)
    clean_df, X_raw, X, y, groups = prepare_dataset(df)
    cv, cv_name = choose_cv(groups)

    print("\nUsing CV:", cv_name)
    print("Rows used:", len(clean_df))

    spaces = build_search_spaces()
    all_results = {}
    best_name = None
    best_estimator = None
    best_mae = float("inf")

    model_dir = Path(args.out_model).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, (model, param_dist) in spaces.items():
        est, result = tune_one_model(
            name=name,
            model=model,
            param_dist=param_dist,
            X=X,
            y=y,
            cv=cv,
            groups=groups,
            n_iter=args.n_iter,
        )
        result["cv_strategy"] = cv_name
        all_results[name] = result

        # fit กับข้อมูลทั้งหมดแล้ว save แยกแต่ละโมเดล
        fitted_est = clone(est)
        fitted_est.fit(X, y)

        model_path = model_dir / f"{name}.joblib"
        joblib.dump(fitted_est, model_path)
        print(f"Saved model -> {model_path}")

        if result["cv_mae"] < best_mae:
            best_mae = result["cv_mae"]
            best_name = name
            best_estimator = clone(est)

    best_estimator.fit(X, y)
    report = {
        "n_rows": int(len(clean_df)),
        "feature_columns_raw": list(X_raw.columns),
        "feature_columns_engineered": list(X.columns),
        "best_model_name": best_name,
        "selection_metric": "cv_mae",
        "best_cv_mae": float(best_mae),
        "cv_strategy": cv_name,
        "all_results": all_results,
        "note": "Tuned regression with randomized search.",
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_estimator, args.out_model)
    Path(args.out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    best_result = all_results[best_name]
    print("\n=== BEST MODEL ===")
    print("Best model:", best_name)
    print("Best CV MAE:", round(best_result["cv_mae"], 4))
    print("Best CV RMSE:", round(best_result["cv_rmse"], 4))
    print("Best CV Accuracy (±1 level):", round(best_result["cv_accuracy_pm1"], 4))
    print("Best CV Accuracy (exact):", round(best_result["cv_accuracy_exact"], 4))
    print("Saved model ->", args.out_model)
    print("Saved report ->", args.out_report)


if __name__ == "__main__":
    main()