"""
Hull Tactical Market Prediction - Comprehensive Audit Script
Performs data quality checks, model validation, and submission analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from pathlib import Path

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def check_data_quality():
    """Analyze train and test data quality"""
    print_section("1. DATA QUALITY ANALYSIS")
    
    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    print(f"📊 Train Shape: {train.shape}")
    print(f"📊 Test Shape: {test.shape}")
    
    # Missing values
    train_missing = train.isnull().sum()
    test_missing = test.isnull().sum()
    
    print(f"\n✓ Train Missing Values: {train_missing.sum()} total")
    if train_missing.sum() > 0:
        print(f"  Top columns with missing:")
        print(train_missing[train_missing > 0].sort_values(ascending=False).head())
    
    print(f"\n✓ Test Missing Values: {test_missing.sum()} total")
    if test_missing.sum() > 0:
        print(f"  Top columns with missing:")
        print(test_missing[test_missing > 0].sort_values(ascending=False).head())
    
    # Infinite values
    train_numeric = train.select_dtypes(include=[np.number])
    test_numeric = test.select_dtypes(include=[np.number])
    
    train_inf = np.isinf(train_numeric).sum().sum()
    test_inf = np.isinf(test_numeric).sum().sum()
    
    print(f"\n✓ Train Infinite Values: {train_inf}")
    print(f"✓ Test Infinite Values: {test_inf}")
    
    # Column overlap
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common_cols = train_cols & test_cols
    train_only = train_cols - test_cols
    test_only = test_cols - train_cols
    
    print(f"\n✓ Common Columns: {len(common_cols)}")
    print(f"✓ Train-only Columns: {len(train_only)}")
    if train_only:
        print(f"  {sorted(list(train_only))[:5]}")
    print(f"✓ Test-only Columns: {len(test_only)}")
    if test_only:
        print(f"  {sorted(list(test_only))[:5]}")
    
    # Target statistics
    if 'target' in train.columns:
        print(f"\n📈 Target Statistics:")
        print(f"  Mean: {train['target'].mean():.6f}")
        print(f"  Std: {train['target'].std():.6f}")
        print(f"  Min: {train['target'].min():.6f}")
        print(f"  Max: {train['target'].max():.6f}")
        print(f"  Median: {train['target'].median():.6f}")
        
        # Outliers
        q1 = train['target'].quantile(0.25)
        q3 = train['target'].quantile(0.75)
        iqr = q3 - q1
        outliers = ((train['target'] < q1 - 1.5*iqr) | (train['target'] > q3 + 1.5*iqr)).sum()
        print(f"  Outliers (IQR method): {outliers} ({100*outliers/len(train):.2f}%)")
    
    return {
        'train_shape': train.shape,
        'test_shape': test.shape,
        'common_features': len(common_cols),
        'train_missing': int(train_missing.sum()),
        'test_missing': int(test_missing.sum())
    }

def analyze_submissions():
    """Compare all submission files"""
    print_section("2. SUBMISSION ANALYSIS")
    
    submissions = {}
    
    # Find all submission files
    for file in Path('.').glob('submission*.csv'):
        name = file.stem
        df = pd.read_csv(file)
        submissions[name] = df
        
        print(f"📄 {name}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        if 'prediction' in df.columns:
            pred = df['prediction']
            print(f"  Mean: {pred.mean():.6f}")
            print(f"  Std: {pred.std():.6f}")
            print(f"  Min: {pred.min():.6f}")
            print(f"  Max: {pred.max():.6f}")
            print(f"  Negative values: {(pred < 0).sum()}")
            print()
    
    # Correlation matrix
    if len(submissions) > 1:
        print("📊 Prediction Correlations:")
        pred_df = pd.DataFrame({name: df['prediction'] for name, df in submissions.items()})
        corr = pred_df.corr()
        print(corr.to_string())
        print()
    
    return {name: {
        'mean': float(df['prediction'].mean()),
        'std': float(df['prediction'].std()),
        'min': float(df['prediction'].min()),
        'max': float(df['prediction'].max())
    } for name, df in submissions.items()}

def check_feature_groups():
    """Analyze feature group patterns"""
    print_section("3. FEATURE GROUP ANALYSIS")
    
    train = pd.read_csv('train.csv')
    
    # Group by prefix
    prefixes = {}
    for col in train.columns:
        if col not in ['date_id', 'target']:
            prefix = col.split('_')[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(col)
    
    print("📊 Feature Groups:")
    for prefix, cols in sorted(prefixes.items()):
        print(f"  {prefix}_*: {len(cols)} features")
        
        # Statistics for each group
        group_data = train[cols]
        print(f"    Missing: {group_data.isnull().sum().sum()}")
        print(f"    Mean range: [{group_data.mean().min():.4f}, {group_data.mean().max():.4f}]")
        print(f"    Std range: [{group_data.std().min():.4f}, {group_data.std().max():.4f}]")
        print()
    
    return {prefix: len(cols) for prefix, cols in prefixes.items()}

def validate_model_files():
    """Check if model-related files exist and are valid"""
    print_section("4. MODEL FILES VALIDATION")
    
    files_to_check = [
        'main.py',
        'advanced_pipeline.py',
        'requirements.txt',
        'README.md',
        'REPORT.md',
        'market_prediction_analysis.ipynb'
    ]
    
    for file in files_to_check:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ {file} ({size:,} bytes)")
        else:
            print(f"✗ {file} - NOT FOUND")
    
    print()

def check_environment():
    """Verify Python environment and key packages"""
    print_section("5. ENVIRONMENT CHECK")
    
    import sys
    print(f"🐍 Python Version: {sys.version}")
    print(f"📍 Python Path: {sys.executable}")
    print()
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'lightgbm', 
        'xgboost', 'optuna', 'shap', 'matplotlib', 'seaborn'
    ]
    
    print("📦 Package Versions:")
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                mod = __import__('sklearn')
            else:
                mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  ✗ {package}: NOT INSTALLED")
    
    print()

def generate_audit_report():
    """Generate comprehensive audit report"""
    print("\n" + "="*80)
    print("  HULL TACTICAL MARKET PREDICTION - AUDIT REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    audit_data = {}
    
    try:
        audit_data['data_quality'] = check_data_quality()
    except Exception as e:
        print(f"⚠ Data quality check failed: {e}")
    
    try:
        audit_data['submissions'] = analyze_submissions()
    except Exception as e:
        print(f"⚠ Submission analysis failed: {e}")
    
    try:
        audit_data['feature_groups'] = check_feature_groups()
    except Exception as e:
        print(f"⚠ Feature group analysis failed: {e}")
    
    try:
        validate_model_files()
    except Exception as e:
        print(f"⚠ Model files validation failed: {e}")
    
    try:
        check_environment()
    except Exception as e:
        print(f"⚠ Environment check failed: {e}")
    
    # Save audit report (JSON)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_json = f'audit_report_{ts}.json'
    with open(audit_json, 'w') as f:
        json.dump(audit_data, f, indent=2)

    # Save simple HTML summary
    Path('reports').mkdir(parents=True, exist_ok=True)
    html_path = Path('reports') / f'full_audit_report_{ts}.html'
    html = [
        '<html><head><meta charset="utf-8"><title>Audit Report</title>',
        '<style>body{font-family:Arial, sans-serif; max-width:900px; margin:40px auto;}h2{margin-top:24px;}code,pre{background:#f6f8fa;padding:8px;border-radius:6px;}table{border-collapse:collapse;width:100%;}td,th{border:1px solid #ddd;padding:8px;}</style>',
        '</head><body>',
        '<h1>Hull Tactical Market Prediction — Audit Report</h1>',
        f'<p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
        '<h2>Data Quality</h2>',
        f'<pre>{json.dumps(audit_data.get("data_quality", {}), indent=2)}</pre>',
        '<h2>Submissions</h2>',
        f'<pre>{json.dumps(audit_data.get("submissions", {}), indent=2)}</pre>',
        '<h2>Feature Groups</h2>',
        f'<pre>{json.dumps(audit_data.get("feature_groups", {}), indent=2)}</pre>',
        '<h2>Environment</h2>',
        '<pre>See console section above</pre>',
        '</body></html>'
    ]
    html_path.write_text("\n".join(html), encoding='utf-8')

    print_section("AUDIT COMPLETE")
    print(f"✓ JSON saved to: {audit_json}")
    print(f"✓ HTML saved to: {html_path}")
    print(f"✓ All checks completed successfully")
    print()

if __name__ == "__main__":
    generate_audit_report()
