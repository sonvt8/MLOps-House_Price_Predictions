"""
Train + CV + MLflow logging, đọc config đúng cấu trúc:

model:
  best_model: RandomForest
  feature_sets:
    rfe: [sqft, bedrooms, bathrooms, location, year_built, condition, house_age, price_per_sqft, bed_bath_ratio]
  parameters: { n_estimators: 200, max_depth: null }
  target_variable: price
  name: house_price_model
  rmse: ..., mae: ..., r2_score: ...

Run:
  python src/models/train_model.py \
    --config configs/model_config.yaml \
    --data data/featured/featured_house_data.csv \
    --models-dir models/trained \
    --mlflow-tracking-uri http://localhost:5555
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib, yaml, mlflow, mlflow.sklearn
import numpy as np, pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .pipeline import build_preprocess, build_model, build_pipeline


# --------------------------- utils --------------------------- #
def setup_logging() -> None:
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    return num, cat


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def select_features_by_config(
    df: pd.DataFrame, model_cfg: Dict, target: str
) -> pd.DataFrame:
    """
    Dùng danh sách 'model.feature_sets.rfe' nếu có; nếu không, dùng toàn bộ cột.
    """
    fs = (model_cfg.get("feature_sets") or {}).get("rfe")
    if not fs:
        return df
    # Giữ đúng thứ tự người dùng cấu hình, bỏ cột thiếu
    cols = [c for c in fs if c in df.columns]
    if target not in cols:
        cols.append(target)
    missing = set(fs) - set(cols)
    if missing:
        logging.warning(f"Missing features from config (ignored): {sorted(missing)}")
    return df[cols]


def pick_model_and_grid(model_cfg: Dict) -> tuple[str, Dict, Dict]:
    """
    Trả về (model_name, base_params, grid).
    YAML của bạn không có 'gridsearch'; ta dùng default nhỏ gọn.
    """
    name = model_cfg.get("best_model") or "RandomForest"
    base_params = model_cfg.get("parameters") or {}
    grid = {
        "model__n_estimators": [base_params.get("n_estimators", 200), 300],
        "model__max_depth": [base_params.get("max_depth", None), 10, 20],
    }
    return name, base_params, grid


def feature_names_after_fit(
    pipe: Pipeline, num_cols: List[str], cat_cols: List[str]
) -> List[str]:
    names = []
    try:
        names.extend(
            pipe.named_steps["preprocess"]
            .named_transformers_["cat"]
            .get_feature_names_out(cat_cols)
            .tolist()
        )
    except Exception:
        pass
    names.extend(num_cols)
    return names


# --------------------------- core --------------------------- #
def train_main(
    config_path: Path, data_path: Path, models_dir: Path, tracking_uri: Optional[str]
) -> None:
    cfg = load_config(config_path)
    model_cfg: Dict = cfg.get("model", {})

    target = model_cfg.get("target_variable") or "price"
    scoring = model_cfg.get("scoring") or "neg_root_mean_squared_error"
    cv = int(cfg.get("cv", 5))

    logging.info(f"Loading featured data: {data_path}")
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    # Lấy đúng danh sách features rfe từ file config
    df = select_features_by_config(df, model_cfg, target)

    # Tách X, y
    y = df[target].astype(float).values
    X = df.drop(columns=[target])

    # Detect kiểu cột
    num_cols, cat_cols = infer_feature_types(df, target)
    logging.info(
        f"Using features from config: {len(num_cols)} numeric, {len(cat_cols)} categorical."
    )

    # Build pipeline + grid
    preprocess = build_preprocess(num_cols, cat_cols)
    model_name, base_params, grid = pick_model_and_grid(model_cfg)
    model = build_model(model_name, base_params)
    pipe = build_pipeline(preprocess, model)

    # GridSearchCV trên toàn pipeline (no leakage)
    logging.info(f"GridSearchCV(model={model_name}, scoring='{scoring}', cv={cv})")
    gs = GridSearchCV(
        pipe, param_grid=grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True, verbose=1
    )
    gs.fit(X, y)
    best_pipe: Pipeline = gs.best_estimator_
    logging.info(f"Best params: {gs.best_params_}")

    # Đánh giá nhanh (trên full data sau refit).
    y_pred = best_pipe.predict(X)
    metrics = evaluate(y, y_pred)
    logging.info(f"Metrics: {metrics}")

    # Lưu artifacts cục bộ
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, models_dir / "model_pipeline.joblib")
    (models_dir / "feature_names.json").write_text(
        json.dumps(
            feature_names_after_fit(best_pipe, num_cols, cat_cols),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (models_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # MLflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        mlflow.log_param("target", target)
        mlflow.log_param("model_name", model_name)
        for k, v in (gs.best_params_ or {}).items():
            mlflow.log_param(k, v)
        # log lại params gốc từ config cho traceability
        for k, v in (model_cfg.get("parameters") or {}).items():
            mlflow.log_param(f"cfg_{k}", v)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_pipe, artifact_path="model")
        # log thêm các file cục bộ
        mlflow.log_artifact(str(models_dir / "model_pipeline.joblib"))
        mlflow.log_artifact(str(models_dir / "feature_names.json"))
        mlflow.log_artifact(str(models_dir / "metrics.json"))

    logging.info("Training & MLflow logging completed.")


# --------------------------- cli --------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train model with GridSearchCV & MLflow (config under 'model')."
    )
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--data", required=True, type=str)
    p.add_argument("--models-dir", required=True, type=str)
    p.add_argument("--mlflow-tracking-uri", default=None, type=str)
    return p.parse_args()


def main() -> None:
    setup_logging()
    a = parse_args()
    train_main(
        Path(a.config).expanduser(),
        Path(a.data).expanduser(),
        Path(a.models_dir).expanduser(),
        a.mlflow_tracking_uri,
    )


if __name__ == "__main__":
    main()
