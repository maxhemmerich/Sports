"""
model.py — XGBoost regressors trained on historical player game logs.

Supports targets: pts, reb, ast, fg3m, blk, stl, tov
Models saved as: model_pts.pkl, model_reb.pkl, model_ast.pkl, etc.

Usage:
    python model.py                    # train all models
    python model.py --target pts       # train only points model
    python model.py --eval             # train + walk-forward CV
    python model.py --retrain          # force retrain even if .pkl exists
"""

import argparse
import joblib
import numpy as np
import os
import pandas as pd
import time
from pathlib import Path

MAX_MODEL_AGE_DAYS = int(os.getenv("MAX_MODEL_AGE_DAYS", "7"))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

from features import build_feature_matrix, MARKET_CONFIG, TARGET_COL

MODEL_DIR = Path(".")
DATA_DIR = Path("data")

# Legacy alias
MODEL_PATH = MODEL_DIR / "model_pts.pkl"

_TARGET_TO_FILE = {
    "pts":  MODEL_DIR / "model_pts.pkl",
    "reb":  MODEL_DIR / "model_reb.pkl",
    "ast":  MODEL_DIR / "model_ast.pkl",
    "fg3m": MODEL_DIR / "model_fg3m.pkl",
    "blk":  MODEL_DIR / "model_blk.pkl",
    "stl":  MODEL_DIR / "model_stl.pkl",
    "tov":  MODEL_DIR / "model_tov.pkl",
}


def _model_path(target: str = "pts") -> Path:
    return _TARGET_TO_FILE.get(target, MODEL_DIR / f"model_{target}.pkl")


def load_training_data(target: str = "pts") -> tuple[pd.DataFrame, pd.Series]:
    """
    Load feature matrix for a given target stat.
    target: 'pts', 'reb', or 'ast'
    """
    feat_cols = None
    for market, (cols, tgt) in MARKET_CONFIG.items():
        if tgt == target:
            feat_cols = cols
            break
    if feat_cols is None:
        raise ValueError(f"Unknown target '{target}'. Choose from: {list({tgt for _, (_, tgt) in MARKET_CONFIG.items()})}")

    df = build_feature_matrix()
    available_features = [c for c in feat_cols if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in feature matrix")
    df_clean = df[available_features + [target]].dropna()
    X = df_clean[available_features]
    y = df_clean[target]
    return X, y


def train(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """Train XGBoost regressor and return fitted model."""
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def evaluate(model: XGBRegressor, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Walk-forward cross-validation using TimeSeriesSplit.
    Returns dict with mean MAE and RMSE.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        m = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        mae = mean_absolute_error(y_te, preds)
        rmse = root_mean_squared_error(y_te, preds)
        maes.append(mae)
        rmses.append(rmse)
        print(f"  Fold {fold + 1}: MAE={mae:.2f}  RMSE={rmse:.2f}")

    return {"mean_mae": np.mean(maes), "mean_rmse": np.mean(rmses)}


def save_model(model: XGBRegressor, path: Path = MODEL_PATH) -> None:
    joblib.dump(model, path)
    print(f"[model] Saved → {path}")


def is_model_stale(target: str = "pts") -> bool:
    """
    Return True if the model should be retrained:
      - .pkl is missing
      - .pkl is older than MAX_MODEL_AGE_DAYS
      - trained feature names don't match current FEATURE_COLS for this target
    """
    path = _model_path(target)
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400
    if age_days > MAX_MODEL_AGE_DAYS:
        return True
    try:
        model = joblib.load(path)
        trained = set(model.get_booster().feature_names or [])
        for market, (feat_cols, tgt) in MARKET_CONFIG.items():
            if tgt == target:
                return trained != set(feat_cols)
    except Exception:
        return True
    return False


def load_model(path: Path | None = None, target: str = "pts") -> XGBRegressor:
    """Load a trained model. Pass path explicitly or resolve from target name."""
    if path is None:
        path = _model_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python model.py --target {target}` first."
        )
    return joblib.load(path)


def predict(features: dict, model: XGBRegressor | None = None, target: str = "pts") -> float:
    """
    Predict stat for a single player-game given a feature dict.

    Args:
        features: dict returned by features.build_live_features()
        model: pre-loaded model (optional, loads from disk if None)
        target: stat column name (pts/reb/ast/fg3m/blk/stl/tov) — used to load the correct model if model is None

    Returns:
        Predicted value (float)
    """
    if model is None:
        model = load_model(target=target)

    trained_features = model.get_booster().feature_names
    if trained_features:
        available = [c for c in trained_features if c in features]
    else:
        # Fallback: use market config for target
        for market, (cols, tgt) in MARKET_CONFIG.items():
            if tgt == target:
                available = [c for c in cols if c in features]
                break
        else:
            available = list(features.keys())

    if not available:
        raise ValueError("No matching feature columns found in input dict.")

    X = pd.DataFrame([features])[available]
    return float(model.predict(X)[0])


def feature_importance(model: XGBRegressor, feature_names: list[str]) -> pd.DataFrame:
    """Return feature importances as a sorted DataFrame."""
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _train_target(target: str, do_eval: bool, retrain: bool) -> XGBRegressor:
    """Train (or load) model for a single target. Returns the trained model."""
    path = _model_path(target)
    if path.exists() and not retrain:
        print(f"[model] {path} already exists — skipping (use --retrain to force).")
        return load_model(target=target)

    print(f"\n=== Training model: {target} ===")
    X, y = load_training_data(target)
    print(f"  Rows: {len(X)}  |  Features: {list(X.columns)}")
    print(f"  Target mean={y.mean():.2f}  std={y.std():.2f}")

    if do_eval:
        print("  --- Walk-Forward CV ---")
        metrics = evaluate(None, X, y)
        print(f"  CV Mean MAE={metrics['mean_mae']:.2f}  RMSE={metrics['mean_rmse']:.2f}")

    model = train(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)

    # OOS sigma via last TimeSeriesSplit fold — more realistic than in-sample residuals.
    # In-sample residuals underestimate true uncertainty due to overfitting.
    oos_sigma = None
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        folds = list(tscv.split(X))
        last_train_idx, last_test_idx = folds[-1]
        m_oos = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0,
        )
        m_oos.fit(X.iloc[last_train_idx], y.iloc[last_train_idx])
        oos_preds = m_oos.predict(X.iloc[last_test_idx])
        oos_sigma = float(np.std(y.iloc[last_test_idx].values - oos_preds))
    except Exception as _e:
        print(f"  [model] OOS sigma failed ({_e}) — falling back to in-sample")

    residual_sigma = oos_sigma if oos_sigma else float(np.std(y.values - preds))
    # Store sigma and MAE on the model so screener can use data-driven thresholds
    model.residual_sigma_ = residual_sigma
    model.train_mae_ = float(mae)
    save_model(model, path)
    print(f"  In-sample MAE={mae:.2f}  RMSE={rmse:.2f}  residual_sigma={residual_sigma:.2f} ({'OOS' if oos_sigma else 'in-sample'})")

    imp_df = feature_importance(model, list(X.columns))
    print(imp_df.to_string(index=False))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA props XGBoost models")
    parser.add_argument("--eval", action="store_true", help="Run walk-forward CV evaluation")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if .pkl exists")
    parser.add_argument(
        "--target",
        choices=["pts", "reb", "ast", "fg3m", "blk", "stl", "tov", "all"],
        default="all",
        help="Which target to train (default: all)",
    )
    args = parser.parse_args()

    targets = ["pts", "reb", "ast", "fg3m", "blk", "stl", "tov"] if args.target == "all" else [args.target]
    for tgt in targets:
        _train_target(tgt, do_eval=args.eval, retrain=args.retrain)

    print("\n[model.py] Done.")
