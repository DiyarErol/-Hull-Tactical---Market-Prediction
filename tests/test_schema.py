import pandas as pd


def test_train_test_column_overlap():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common = train_cols & test_cols

    # Expect a reasonable number of common features
    assert len(common) >= 90, f"Common columns too few: {len(common)}"

    # Known train-only vs test-only differences should exist
    assert "market_forward_excess_returns" in train_cols, "Target missing in train.csv"
    assert "is_scored" in test_cols, "is_scored should be present in test.csv"
