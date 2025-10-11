from __future__ import annotations

import logging
from collections.abc import Callable

from sklearn.compose import ColumnTransformer

# Core sklearn regressors
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


def build_preprocess(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    return preprocess


def _optional_imports() -> dict[str, Callable]:
    """Return optional model constructors if libs are installed."""
    registry: dict[str, Callable] = {}
    # XGBoost
    try:
        from xgboost import XGBRegressor  # type: ignore

        registry.update(
            {
                "xgboost": XGBRegressor,
                "xgb": XGBRegressor,
                "xgbregressor": XGBRegressor,
            }
        )
    except Exception:
        pass
    # LightGBM
    try:
        from lightgbm import LGBMRegressor  # type: ignore

        registry.update(
            {
                "lightgbm": LGBMRegressor,
                "lgbm": LGBMRegressor,
                "lgbmregressor": LGBMRegressor,
            }
        )
    except Exception:
        pass
    # CatBoost
    try:
        from catboost import CatBoostRegressor  # type: ignore

        registry.update(
            {
                "catboost": CatBoostRegressor,
                "catboostregressor": CatBoostRegressor,
            }
        )
    except Exception:
        pass
    return registry


def _base_registry() -> dict[str, Callable]:
    """Always-available sklearn regressors."""
    return {
        # Tree-based
        "randomforest": RandomForestRegressor,
        "random_forest": RandomForestRegressor,
        "rf": RandomForestRegressor,
        "gradientboosting": GradientBoostingRegressor,
        "gbr": GradientBoostingRegressor,
        "extra_trees": ExtraTreesRegressor,
        "extratrees": ExtraTreesRegressor,
        # Linear family
        "linear": LinearRegression,
        "linearregression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elasticnet": ElasticNet,
        # Kernel-based
        "svr": SVR,
    }


def _get_model_registry() -> dict[str, Callable]:
    reg = _base_registry()
    reg.update(_optional_imports())
    return reg


def build_model(name: str, params: dict | None = None):
    """Build a regressor dynamically by name.
    Supported aliases: rf/randomforest, gbr, extratrees, linear/ridge/lasso/elasticnet, svr,
    and (if installed) xgboost/xgb, lgbm/lightgbm, catboost.
    """
    params = params or {}
    name_low = (name or "").lower().strip()
    registry = _get_model_registry()

    if name_low in registry:
        ctor = registry[name_low]
        return ctor(**params)

    logging.warning(
        f"[build_model] Unsupported model name '{name}'. Falling back to RandomForestRegressor."
    )
    return RandomForestRegressor(random_state=42, n_jobs=-1, **params)


def build_pipeline(preprocess, model) -> Pipeline:
    return Pipeline([("preprocess", preprocess), ("model", model)])
