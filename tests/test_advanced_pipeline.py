"""
Unit tests for advanced_pipeline helper functions.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_pipeline import calculate_metrics, add_technical_indicators


def test_calculate_metrics_basic():
    """Test basic metric calculation"""
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    y_pred = np.array([0.015, -0.015, 0.025, -0.005, 0.018])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'R2' in metrics
    assert 'Direction_Accuracy' in metrics
    assert metrics['RMSE'] > 0
    assert 0 <= metrics['Direction_Accuracy'] <= 1


def test_calculate_metrics_perfect():
    """Test metrics with perfect predictions"""
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = y_true.copy()
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics['RMSE'] == pytest.approx(0, abs=1e-10)
    assert metrics['MAE'] == pytest.approx(0, abs=1e-10)
    assert metrics['R2'] == pytest.approx(1.0, abs=1e-10)
    assert metrics['Direction_Accuracy'] == 1.0


def test_add_technical_indicators_shape():
    """Test that technical indicators are added properly"""
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'date_id': range(100),
        'D_1': np.random.randn(100),
        'D_2': np.random.randn(100),
        'M_1': np.random.randn(100),
        'M_2': np.random.randn(100),
    })
    
    feature_cols = ['D_1', 'D_2', 'M_1', 'M_2']
    result = add_technical_indicators(df, feature_cols)
    
    # Should have more columns than input
    assert result.shape[1] > df.shape[1]
    assert result.shape[0] == df.shape[0]
    
    # Check for expected new columns
    assert any('rolling_mean' in col for col in result.columns)
    assert any('rsi' in col for col in result.columns)
    assert any('macd' in col for col in result.columns)


def test_add_technical_indicators_no_inf():
    """Test that technical indicators don't produce inf values"""
    np.random.seed(42)
    df = pd.DataFrame({
        'P_1': np.random.randn(50),
        'P_2': np.random.randn(50),
    })
    
    result = add_technical_indicators(df, ['P_1', 'P_2'])
    
    # Check no inf values (NaN is acceptable for initial windows)
    assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()
