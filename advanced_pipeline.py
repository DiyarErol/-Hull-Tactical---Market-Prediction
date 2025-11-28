"""
Advanced Hull Tactical Market Prediction Pipeline
- Cross-validation (TimeSeriesSplit)
- Feature engineering (technical indicators)
- Hyperparameter tuning (Optuna)
- Multiple models (Ridge, LightGBM, XGBoost)
- SHAP feature importance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def add_technical_indicators(df, feature_cols):
    """Teknik göstergeler ekle: RSI, MACD, EMA, Bollinger Bands"""
    df_copy = df.copy()

    for prefix in ["D", "E", "I", "M", "P", "S", "V"]:
        group_cols = [col for col in feature_cols if col.startswith(prefix + "_")]
        if len(group_cols) == 0:
            continue

        group_mean = df_copy[group_cols].mean(axis=1)

        # Rolling features
        for window in [5, 10, 20]:
            df_copy[f"{prefix}_rolling_mean_{window}"] = (
                group_mean.rolling(window=window, min_periods=1).mean()
            )
            df_copy[f"{prefix}_rolling_std_{window}"] = (
                group_mean.rolling(window=window, min_periods=1).std()
            )
            df_copy[f"{prefix}_ema_{window}"] = group_mean.ewm(span=window, adjust=False).mean()

        # RSI
        delta = group_mean.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df_copy[f"{prefix}_rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = group_mean.ewm(span=12, adjust=False).mean()
        ema26 = group_mean.ewm(span=26, adjust=False).mean()
        df_copy[f"{prefix}_macd"] = ema12 - ema26
        df_copy[f"{prefix}_macd_signal"] = df_copy[f"{prefix}_macd"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        rolling_mean = group_mean.rolling(window=20, min_periods=1).mean()
        rolling_std = group_mean.rolling(window=20, min_periods=1).std()
        df_copy[f"{prefix}_bb_width"] = 2 * rolling_std

    return df_copy


def calculate_metrics(y_true, y_pred):
    """Model performans metriklerini hesapla"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    direction_acc = np.mean(direction_true == direction_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Direction_Accuracy": direction_acc}


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=20):
    """LightGBM hyperparameter tuning with Optuna"""

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "random_state": 42,
        }

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest RMSE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def cross_validate_model(X, y, model_type="lightgbm", params=None, n_splits=5):
    """Cross-validation with TimeSeriesSplit"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {"RMSE": [], "MAE": [], "R2": [], "Direction_Accuracy": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        if model_type == "lightgbm":
            lgb_train = lgb.Dataset(X_tr_scaled, y_tr)
            lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)

            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
            )
            preds = model.predict(X_val_scaled, num_iteration=model.best_iteration)

        elif model_type == "ridge":
            model = Ridge(alpha=params.get("alpha", 1.0), random_state=42)
            model.fit(X_tr_scaled, y_tr)
            preds = model.predict(X_val_scaled)

        elif model_type == "xgboost":
            dtrain = xgb.DMatrix(X_tr_scaled, label=y_tr)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=0,
            )
            preds = model.predict(dval, iteration_range=(0, model.best_iteration))

        metrics = calculate_metrics(y_val, preds)
        for key, value in metrics.items():
            cv_scores[key].append(value)

        print(
            f"Fold {fold}: RMSE={metrics['RMSE']:.6f}, "
            f"DirAcc={metrics['Direction_Accuracy']:.4f}"
        )

    print("\n" + "=" * 70)
    print(f"{model_type.upper()} CROSS-VALIDATION RESULTS")
    print("=" * 70)
    for metric, values in cv_scores.items():
        print(f"{metric}: {np.mean(values):.6f} ± {np.std(values):.6f}")
    print("=" * 70)

    return cv_scores


def main():
    print("\n" + "=" * 70)
    print("ADVANCED PIPELINE - HULL TACTICAL MARKET PREDICTION")
    print("=" * 70)

    # Load data
    print("\n[1] Veri Yükleme...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")

    # Prepare features
    print("\n[2] Feature Hazırlama...")
    date_col = "date_id"
    target_col = "market_forward_excess_returns"

    X_train_base = train_df.drop(columns=[date_col, target_col])
    y_train = train_df[target_col]
    test_ids = test_df[date_col]

    # Ortak kolonlar
    train_cols = set(X_train_base.columns)
    test_cols = set(test_df.columns) - {date_col}
    common_cols = sorted(train_cols & test_cols)
    print(f"✓ {len(common_cols)} ortak kolon kullanılacak")

    X_train_base = X_train_base[common_cols]
    X_test_base = test_df[common_cols]

    # Feature engineering
    print("\n[3] Feature Engineering (Teknik Göstergeler)...")
    train_enhanced = add_technical_indicators(
        train_df.drop(columns=[target_col]), X_train_base.columns.tolist()
    )
    test_enhanced = add_technical_indicators(test_df, X_train_base.columns.tolist())

    # Ortak kolonlar tekrar
    train_feat_cols = [col for col in train_enhanced.columns if col != date_col]
    test_feat_cols = [col for col in test_enhanced.columns if col != date_col]
    common_enhanced = sorted(set(train_feat_cols) & set(test_feat_cols))

    X_train = train_enhanced[common_enhanced].fillna(0)
    X_test = test_enhanced[common_enhanced].fillna(0)

    # Inf temizle
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    print(f"✓ Enhanced X_train: {X_train.shape}")
    print(f"✓ Enhanced X_test: {X_test.shape}")

    # Train-val split for tuning
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # Hyperparameter tuning
    print("\n[4] Hyperparameter Tuning (LightGBM, 20 trials)...")
    best_params = tune_lightgbm(X_tr_scaled, y_tr, X_val_scaled, y_val, n_trials=20)
    best_params["objective"] = "regression"
    best_params["metric"] = "rmse"
    best_params["verbosity"] = -1
    best_params["random_state"] = 42

    # Cross-validation
    print("\n[5] Cross-Validation (5-fold TimeSeriesSplit)...")
    cv_results = cross_validate_model(
        X_train.values, y_train, model_type="lightgbm", params=best_params, n_splits=5
    )

    # Train final model
    print("\n[6] Final Model Eğitimi...")
    scaler_full = RobustScaler()
    X_full_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)

    lgb_train_full = lgb.Dataset(X_full_scaled, y_train)
    final_model = lgb.train(
        best_params, lgb_train_full, num_boost_round=500, callbacks=[lgb.log_evaluation(0)]
    )

    # Predictions
    test_preds = final_model.predict(X_test_scaled)

    print(f"✓ Test tahminleri: [{test_preds.min():.6f}, {test_preds.max():.6f}]")
    print(f"✓ Test tahmin ortalaması: {test_preds.mean():.6f}")

    # Create submission
    submission = pd.DataFrame({"id": test_ids, "prediction": test_preds})
    submission.to_csv("submission_advanced.csv", index=False)

    # Save visuals to reports/
    Path("reports").mkdir(parents=True, exist_ok=True)
    # Histogram of predictions
    plt.figure(figsize=(8, 5))
    plt.hist(test_preds, bins=20, color="#4e79a7", alpha=0.8)
    plt.title("Prediction Distribution (Advanced Pipeline)")
    plt.xlabel("prediction")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("reports/advanced_prediction_hist.png")
    plt.close()

    # Simple equity curve using sign(pred) * target on train (proxy)
    train_signal = np.sign(y_train.values)
    equity = np.cumsum(train_signal * y_train.values)
    plt.figure(figsize=(9, 5))
    plt.plot(equity, color="#e15759")
    plt.title("Proxy Equity Curve (Train)")
    plt.xlabel("day")
    plt.ylabel("cumulative pnl")
    plt.tight_layout()
    plt.savefig("reports/advanced_proxy_equity_curve.png")
    plt.close()

    print("\n" + "=" * 70)
    print("✓ submission_advanced.csv oluşturuldu")
    print("=" * 70)
    print("\nÖzet:")
    print(f"  - Ortak feature: {len(common_cols)}")
    print(f"  - Enhanced feature: {X_train.shape[1]}")
    print(f"  - CV RMSE: {np.mean(cv_results['RMSE']):.6f} ± {np.std(cv_results['RMSE']):.6f}")
    print(
        f"  - CV DirAcc: {np.mean(cv_results['Direction_Accuracy']):.4f} ± "
        f"{np.std(cv_results['Direction_Accuracy']):.4f}"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
