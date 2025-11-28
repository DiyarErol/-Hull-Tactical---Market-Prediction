import os
import pandas as pd


def test_submission_csv_exists_and_format():
    """Ensure submission files exist and have required columns and sizes."""
    files = [
        "submission.csv",
        "submission_advanced.csv",
    ]
    existing = [f for f in files if os.path.exists(f)]
    assert existing, "No submission CSV found. Run main.py or advanced_pipeline.py."

    for f in existing:
        df = pd.read_csv(f)
        assert set(["id", "prediction"]).issubset(df.columns), f"{f} must contain 'id' and 'prediction' columns"
        assert len(df) > 0, f"{f} must contain at least one row"
        # Basic sanity checks
        assert df["prediction"].dtype.kind in "fi", f"{f} prediction must be numeric"
