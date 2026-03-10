"""
model.py — XGBoost regressor trained on historical player game logs.

Target  : pts  (actual points scored)
Features: rolling averages, opponent defense, pace, rest, home/away, travel

Usage:
    python model.py          # train and save model.pkl
    python model.py --eval   # train, evaluate, save
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

from features import build_feature_matrix, FEATURE_COLS, TARGET_COL

MODEL_PATH = Path("model.pkl")
DATA_DIR = Path("data")


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load feature matrix, drop rows with any NaN in feature cols,
    return X (features) and y (target).
    """
    df = build_feature_matrix()
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[available_features + [TARGET_COL]].dropna()
    X = df_clean[available_features]
    y = df_clean[TARGET_COL]
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


def load_model(path: Path = MODEL_PATH) -> XGBRegressor:
    if not path.exists():
        raise FileNotFoundError(
            f"model.pkl not found at {path}. Run `python model.py` first to train."
        )
    return joblib.load(path)


def predict(features: dict, model: XGBRegressor | None = None) -> float:
    """
    Predict points for a single player-game given a feature dict.

    Args:
        features: dict returned by features.build_live_features()
        model: pre-loaded model (optional, loads from disk if None)

    Returns:
        Predicted points (float)
    """
    if model is None:
        model = load_model()

    # Use the exact feature names the model was trained on (from the booster)
    # This avoids mismatch if FEATURE_COLS and model diverge.
    trained_features = model.get_booster().feature_names
    if trained_features:
        available = [c for c in trained_features if c in features]
    else:
        available = [c for c in FEATURE_COLS if c in features]
    if not available:
        raise ValueError("No matching feature columns found in input dict.")

    X = pd.DataFrame([features])[available]
    return float(model.predict(X)[0])


def feature_importance(model: XGBRegressor, feature_names: list[str]) -> pd.DataFrame:
    """Return feature importances as a sorted DataFrame."""
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA props XGBoost model")
    parser.add_argument("--eval", action="store_true", help="Run walk-forward CV evaluation")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model.pkl exists")
    args = parser.parse_args()

    if MODEL_PATH.exists() and not args.retrain:
        print(f"[model] model.pkl already exists. Use --retrain to force retrain.")
        model = load_model()
    else:
        print("=== Model Training ===")
        X, y = load_training_data()
        print(f"Training set: {len(X)} rows, {X.shape[1]} features")
        print(f"Features: {list(X.columns)}")
        print(f"Target: {TARGET_COL}  |  mean={y.mean():.1f}  std={y.std():.1f}")

        if args.eval:
            print("\n--- Walk-Forward Cross-Validation ---")
            metrics = evaluate(None, X, y)
            print(f"\nCV Results: Mean MAE={metrics['mean_mae']:.2f}  Mean RMSE={metrics['mean_rmse']:.2f}")

        print("\n--- Training Final Model (full data) ---")
        model = train(X, y)
        save_model(model)

        print("\n--- Feature Importance ---")
        imp_df = feature_importance(model, list(X.columns))
        print(imp_df.to_string(index=False))

        # Quick in-sample sanity check
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = root_mean_squared_error(y, preds)
        print(f"\nIn-sample  MAE={mae:.2f}  RMSE={rmse:.2f}")

    print("\n[model.py] Done.")
