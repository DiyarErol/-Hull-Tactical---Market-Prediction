"""
Hull Tactical Market Prediction - Ana Pipeline
Bu script train.csv'yi okur, basit modeller eğitir ve submission.csv oluşturur.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def load_data():
    """Veriyi yükle ve temel kontrolleri yap"""
    print("=" * 70)
    print("VERİ YÜKLEME")
    print("=" * 70)

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    print(f"✓ Train shape: {train_df.shape}")
    print(f"✓ Test shape: {test_df.shape}")

    return train_df, test_df


def prepare_features(train_df, test_df):
    """Feature ve target ayrımı yap"""
    print("\n" + "=" * 70)
    print("FEATURE HAZIRLAMA")
    print("=" * 70)

    date_col = "date_id"
    target_col = "market_forward_excess_returns"

    # Feature ve target ayır
    X_train = train_df.drop(columns=[date_col, target_col])
    y_train = train_df[target_col]
    test_ids = test_df[date_col]

    # Ortak kolonları bul
    train_cols = set(X_train.columns)
    test_cols = set(test_df.columns) - {date_col}
    common_cols = sorted(train_cols & test_cols)

    print(f"✓ Train'de {len(train_cols)} kolon, Test'te {len(test_cols)} kolon")
    print(f"✓ Ortak {len(common_cols)} kolon kullanılacak")

    # Sadece ortak kolonları kullan
    X_train = X_train[common_cols]
    X_test = test_df[common_cols]

    # Eksik değerleri doldur
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ y_train shape: {y_train.shape}")
    print(f"✓ X_test shape: {X_test.shape}")

    return X_train, y_train, X_test, test_ids


def calculate_metrics(y_true, y_pred, prefix=""):
    """Model performans metriklerini hesapla"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Direction Accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    direction_acc = np.mean(direction_true == direction_pred)

    print(f"\n{prefix} Metrics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  Direction Accuracy: {direction_acc:.4f}")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Direction_Accuracy": direction_acc}


def train_models(X_train, y_train):
    """Ridge ve LightGBM modellerini eğit"""
    print("\n" + "=" * 70)
    print("MODEL EĞİTİMİ")
    print("=" * 70)

    # Train-Val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=False
    )

    # Scaling
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_full_scaled = scaler.fit_transform(X_train)

    # 1. Ridge Model
    print("\n[1] Ridge Regression")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_tr_scaled, y_tr)

    ridge_train_pred = ridge_model.predict(X_tr_scaled)
    ridge_val_pred = ridge_model.predict(X_val_scaled)

    calculate_metrics(y_tr, ridge_train_pred, "Ridge Train")
    calculate_metrics(y_val, ridge_val_pred, "Ridge Val")

    # 2. LightGBM Model
    print("\n[2] LightGBM")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
    }

    lgb_train = lgb.Dataset(X_tr_scaled, y_tr)
    lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)],
    )

    lgb_train_pred = lgb_model.predict(X_tr_scaled, num_iteration=lgb_model.best_iteration)
    lgb_val_pred = lgb_model.predict(X_val_scaled, num_iteration=lgb_model.best_iteration)

    calculate_metrics(y_tr, lgb_train_pred, "LightGBM Train")
    lgb_val_metrics = calculate_metrics(y_val, lgb_val_pred, "LightGBM Val")

    # Tüm veri ile final model
    print("\n[3] Final Model (Tüm Train Verisi)")
    ridge_full = Ridge(alpha=1.0, random_state=42)
    ridge_full.fit(X_full_scaled, y_train)

    lgb_full_train = lgb.Dataset(X_full_scaled, y_train)
    lgb_full = lgb.train(lgb_params, lgb_full_train, num_boost_round=lgb_model.best_iteration)

    return ridge_full, lgb_full, scaler, lgb_val_metrics


def create_submission(ridge_model, lgb_model, scaler, X_test, test_ids, lgb_val_metrics):
    """Test tahminlerini oluştur ve submission.csv kaydet"""
    print("\n" + "=" * 70)
    print("TEST TAHMİNLERİ VE SUBMISSION")
    print("=" * 70)

    X_test_scaled = scaler.transform(X_test)

    # Tahminler
    ridge_pred = ridge_model.predict(X_test_scaled)
    lgb_pred = lgb_model.predict(X_test_scaled)

    # Ensemble (LightGBM val performansına göre ağırlıklandır)
    # LightGBM daha iyi performans gösteriyorsa daha fazla ağırlık ver
    lgb_weight = 0.7  # LightGBM için daha yüksek ağırlık
    ridge_weight = 0.3

    ensemble_pred = ridge_weight * ridge_pred + lgb_weight * lgb_pred

    print(f"✓ Ridge predictions: [{ridge_pred.min():.4f}, {ridge_pred.max():.4f}]")
    print(f"✓ LightGBM predictions: [{lgb_pred.min():.4f}, {lgb_pred.max():.4f}]")
    print(f"✓ Ensemble predictions: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")

    # Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "prediction": ensemble_pred})

    submission_df.to_csv("submission.csv", index=False)
    print(f"\n✓ submission.csv oluşturuldu ({len(submission_df)} satır)")

    return submission_df


def main():
    """Ana pipeline"""
    print("\n" + "=" * 70)
    print("HULL TACTICAL MARKET PREDICTION - PIPELINE")
    print("=" * 70)

    # 1. Veri yükle
    train_df, test_df = load_data()

    # 2. Feature hazırla
    X_train, y_train, X_test, test_ids = prepare_features(train_df, test_df)

    # 3. Modelleri eğit
    ridge_model, lgb_model, scaler, lgb_val_metrics = train_models(X_train, y_train)

    # 4. Submission oluştur
    submission_df = create_submission(
        ridge_model, lgb_model, scaler, X_test, test_ids, lgb_val_metrics
    )

    print("\n" + "=" * 70)
    print("PIPELINE TAMAMLANDI")
    print("=" * 70)
    print("\nSonraki adımlar:")
    print("  1. Jupyter Notebook açın: jupyter notebook")
    print("  2. market_prediction_analysis.ipynb dosyasını açın")
    print("  3. Detaylı analiz ve model geliştirme yapın")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
