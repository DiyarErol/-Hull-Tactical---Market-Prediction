# Hull Tactical Market Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI: GitHub Actions](https://github.com/DiyarErol/-Hull-Tactical---Market-Prediction/actions/workflows/python.yml/badge.svg)](https://github.com/DiyarErol/-Hull-Tactical---Market-Prediction/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Machine learning-based market prediction project built for the Kaggle competition. Uses `train.csv` to predict the target `market_forward_excess_returns`.

## 🚀 Quick Start

### Requirements
- Python 3.11.9
- pip 25.3

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the baseline pipeline
python main.py

# 3. Run the advanced pipeline
python advanced_pipeline.py
```

Key files:
- `market_prediction_analysis.ipynb` — analysis notebook
- `requirements.txt` — Python dependencies (~149 packages)

### Main Script (main.py)
- Ridge Regression (baseline)
- LightGBM (gradient boosting)
- Metrics: RMSE, MAE, R², Direction Accuracy

Run:
```bash
python main.py
```

Output: `submission.csv` (id + prediction)

### Advanced Pipeline (advanced_pipeline.py)
Run with options:
```bash
# Default (126-day rolling Sharpe)
python advanced_pipeline.py

# Custom rolling Sharpe window
python advanced_pipeline.py --sharpe_window 252
```

### Ensemble Submission (make_submission.py)
Combine basic and advanced predictions:
```bash
# Default 50/50 blend
python make_submission.py

# Custom weights
python make_submission.py --weight_basic 0.3 --weight_advanced 0.7
```

Output: `submission_ensemble.csv`

### Jupyter Notebook (market_prediction_analysis.ipynb)
Comprehensive analysis and experimentation environment.

Contents:
1. Data exploration: distributions, missing values, feature groups
2. Feature engineering:
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - EMA (Exponential Moving Average)
   - Bollinger Bands
   - Rolling statistics (mean, std)
3. Model training:
   - Ridge Regression
   - LightGBM
   - XGBoost
   - TimeSeriesSplit validation
4. Feature importance:
   - LightGBM gain-based importance
   - SHAP values (TreeExplainer)
5. Ensemble methods: weighted averaging
6. Visualization: histograms, comparison tables, equity curve

## 🔧 Technologies

### Data Processing
- pandas 2.3.3
- numpy 2.3.5
- scipy 1.16.3

### Machine Learning
- scikit-learn 1.7.2
- lightgbm 4.6.0
- xgboost 3.1.2
- optuna 4.6.0 (ready for hyperparameter tuning)

### Model Interpretation
- shap 0.50.0

### Visualization
- matplotlib 3.10.7
- seaborn 0.13.2
- plotly 6.5.0

### Development Tools
- jupyter 1.1.1
- black 25.11.0 (code formatter)
- flake8 7.3.0 (linter)
- pytest 9.0.1 (test framework)

## 📊 Model Performance

Recent validation results:

| Model | RMSE | R² | Direction Acc |
|-------|------|-----|---------------|
| Ridge | 0.0119 | -0.146 | 0.5008 |
| LightGBM | 0.0111 | 0.0017 | 0.5302 |
| **Ensemble** | **0.0109** | **0.008** | **0.525** |

## 🎨 Feature Groups

`train.csv` includes 96 features grouped into 7 families:

- **D_**: Derivative features (~15 cols)
- **E_**: Economic indicators (~10 cols)
- **I_**: Interest rate features (~12 cols)
- **M_**: Market features (~20 cols)
- **P_**: Price features (~18 cols)
- **S_**: Sentiment features (~8 cols)
- **V_**: Volatility features (~13 cols)

## 🛠️ Development Tips

### Add a New Model
```python
# Add into train_models in main.py
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
    # Train model and return RMSE
    ...
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### Feature Engineering
```python
# Add technical indicators
def add_momentum_features(df):
    df['momentum_5'] = df['feature'].pct_change(5)
    df['momentum_10'] = df['feature'].pct_change(10)
    return df
```

## 📝 Notes

- Data leakage: test set has `lagged_*` columns not in train; pipeline automatically selects common columns.
- Scaling: RobustScaler (robust to outliers).
- Time series: TimeSeriesSplit with strict temporal ordering (shuffle=False).
- Early stopping: LightGBM patience 50 rounds.
- Reproducibility: `random_state=42` used consistently.

## 🚧 Roadmap

### Short Term
- [x] Baseline pipeline (Ridge + LightGBM)
- [x] Submission file generation
- [x] Jupyter notebook
- [ ] Cross-validation (5-fold TimeSeriesSplit)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Stacking ensemble

### Medium Term
- [ ] Ablation study (feature group impact)
- [ ] Permutation importance
- [ ] SHAP dependency plots
- [ ] Backtest simulation
- [ ] Error analysis (worst days, regime detection)

### Long Term
- [ ] Neural network (MLP, LSTM)
- [ ] AutoML (FLAML, Auto-sklearn)
- [ ] Feature selection (Boruta, RFE)
- [ ] Model calibration
- [ ] Production deployment (FastAPI)

## 📦 Artifacts (CI)

GitHub Actions CI uploads the following artifacts on each push:
- `submission.csv`, `submission_advanced.csv`
- `audit_report_*.json`
- (If any) `reports/full_audit_report_*.html`
- Visuals: `reports/advanced_prediction_hist.png`, `reports/advanced_proxy_equity_curve.png`, `reports/walkforward_oof_equity.png`

## 🔁 Walk-Forward OOF Backtest

- Transaction cost: 2 bps per day
- Latest run (local):
  - OOF Sharpe: -0.13
  - Max Drawdown: ~0.00
- Artifacts:
  - Equity: `reports/walkforward_oof_equity.png`
  - Metrics JSON: `reports/walkforward_oof_fin_*.json`
  - Rolling Sharpe: `reports/rolling_sharpe_oof.png`

**View all runs:**
```bash
python scripts/summarize_oof.py
```Open Actions → “Python CI” run and download from “Artifacts”.

## 🤝 Contributing

Open an issue for proposals and improvements.

### Contact
- **Email:** [eroldiyar41@gmail.com](mailto:eroldiyar41@gmail.com)
- **LinkedIn:** [Diyar Erol](https://www.linkedin.com/in/diyar-erol-1b3837356/)
- **GitHub Issues:** [Report a bug](https://github.com/DiyarErol/-Hull-Tactical---Market-Prediction/issues)

## 📄 License

This project is for educational purposes.

---

**Last Update:** 28 Nov 2025  
**Python:** 3.11.9 | **pip:** 25.3 | **Packages:** ~149
