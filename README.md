# Hull Tactical Market Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI: GitHub Actions](https://github.com/DiyarErol/-Hull-Tactical---Market-Prediction/actions/workflows/python.yml/badge.svg)](https://github.com/DiyarErol/-Hull-Tactical---Market-Prediction/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Kaggle yarışması için geliştirilmiş makine öğrenmesi tabanlı piyasa tahmin projesi. Train.csv verisini kullanarak `market_forward_excess_returns` hedef değişkenini tahmin eder.

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.11.9
- pip 25.3

### Kurulum

```bash
# 1. Bağımlılıkları yükle

# 2. Basit pipeline çalıştır
python main.py


```
├── market_prediction_analysis.ipynb   # Detaylı analiz notebook'u
├── requirements.txt                   # Python bağımlılıkları (149 paket)

### Ana Script (main.py)
- Ridge Regression (baseline)
- LightGBM (gradient boosting)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Direction Accuracy (yön tahmin doğruluğu)

**Çalıştırma:**
```bash
python main.py
```

**Çıktı:** `submission.csv` (id + prediction)

### Jupyter Notebook (market_prediction_analysis.ipynb)
Kapsamlı analiz ve geliştirme ortamı:

**İçerik:**
1. **Veri Keşfi:** Dağılım analizi, eksik değer kontrolü, feature grupları
2. **Feature Engineering:** 
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - EMA (Exponential Moving Average)
   - Bollinger Bands
   - Rolling istatistikler (mean, std)
3. **Model Eğitimi:**
   - Ridge Regression
   - LightGBM
   - XGBoost
   - TimeSeriesSplit validation
4. **Feature Importance:**
   - LightGBM gain-based importance
   - SHAP values (TreeExplainer)
5. **Ensemble Methods:** Weighted averaging
6. **Görselleştirme:** Dağılım grafikleri, karşılaştırma tabloları

## 🔧 Kullanılan Teknolojiler

### Veri İşleme
- pandas 2.3.3
- numpy 2.3.5
- scipy 1.16.3

### Makine Öğrenmesi
- scikit-learn 1.7.2
- lightgbm 4.6.0
- xgboost 3.1.2
- optuna 4.6.0 (hyperparameter tuning için hazır)

### Model Yorumlama
- shap 0.50.0

### Görselleştirme
- matplotlib 3.10.7
- seaborn 0.13.2
- plotly 6.5.0

### Geliştirme Araçları
- jupyter 1.1.1
- black 25.11.0 (code formatter)
- flake8 7.3.0 (linter)
- pytest 9.0.1 (test framework)

## 📊 Model Performansı

Son çalıştırma sonuçları (validation set):

| Model | RMSE | R² | Direction Acc |
|-------|------|-----|---------------|
| Ridge | 0.0119 | -0.146 | 0.5008 |
| LightGBM | 0.0111 | 0.0017 | 0.5302 |
| **Ensemble** | **0.0109** | **0.008** | **0.525** |

## 🎨 Feature Grupları

Train.csv'de 96 feature, 7 ana gruba ayrılmış:

- **D_** : Derivative features (~15 kolon)
- **E_** : Economic indicators (~10 kolon)
- **I_** : Interest rate features (~12 kolon)
- **M_** : Market features (~20 kolon)
- **P_** : Price features (~18 kolon)
- **S_** : Sentiment features (~8 kolon)
- **V_** : Volatility features (~13 kolon)

## 🛠️ Geliştirme İpuçları

### Yeni Model Ekleme
```python
# main.py içinde train_models fonksiyonuna ekle
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_tr_scaled, y_tr)
rf_pred = rf_model.predict(X_val_scaled)
rf_metrics = calculate_metrics(y_val, rf_pred, "RandomForest Val")
```

### Hyperparameter Tuning
```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
    }
    # Model eğit ve RMSE döndür
    ...
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### Feature Engineering
```python
# Teknik göstergeler ekle
def add_momentum_features(df):
    df['momentum_5'] = df['feature'].pct_change(5)
    df['momentum_10'] = df['feature'].pct_change(10)
    return df
```

## 📝 Notlar

- **Veri Leakage:** Test setinde `lagged_*` kolonları var, train'de yok. Pipeline ortak kolonları otomatik seçer.
- **Scaling:** RobustScaler kullanılıyor (outlier'lara dayanıklı).
- **Time Series:** TimeSeriesSplit ile zamansal bölünme (shuffle=False).
- **Early Stopping:** LightGBM'de 50 round patience.
- **Reproducibility:** random_state=42 her yerde sabit.

## 🚧 Geliştirme Yol Haritası

### Kısa Vade
- [x] Basit pipeline (Ridge + LightGBM)
- [x] Submission dosyası üretimi
- [x] Jupyter notebook hazırlama
- [ ] Cross-validation (5-fold TimeSeriesSplit)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Stacking ensemble

### Orta Vade
- [ ] Ablation study (feature grubu etkisi)
- [ ] Permutation importance
- [ ] SHAP dependency plots
- [ ] Backtest simülasyonu
- [ ] Error analizi (worst days, regime detection)

### Uzun Vade
- [ ] Neural network (MLP, LSTM)
- [ ] AutoML (FLAML, Auto-sklearn)
- [ ] Feature selection (Boruta, RFE)
- [ ] Model calibration
- [ ] Production deployment (FastAPI)

## 📦 Artifacts (CI)

GitHub Actions CI, her push’ta aşağıdaki artifact’leri yükler:
- `submission.csv`, `submission_advanced.csv`
- `audit_report_*.json`
- (Varsa) `reports/full_audit_report_*.html`

Actions → “Python CI” çalıştırmasını açıp “Artifacts” sekmesinden indirebilirsiniz.

## 🤝 Katkı

Öneriler ve iyileştirmeler için issue açabilirsiniz.

### Contact
- **Email:** [eroldiyar41@gmail.com](mailto:eroldiyar41@gmail.com)
- **LinkedIn:** [Diyar Erol](https://www.linkedin.com/in/diyar-erol-1b3837356/)
- **GitHub Issues:** [Report a bug](https://github.com/DiyarErol/-Hull-Tactical---Market-Prediction/issues)

## 📄 Lisans

Bu proje eğitim amaçlıdır.

---

**Son Güncelleme:** 28 Kasım 2025  
**Python:** 3.11.9 | **pip:** 25.3 | **Paketler:** 149
