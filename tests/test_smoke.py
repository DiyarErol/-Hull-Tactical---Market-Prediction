import os
import importlib


def test_data_files_exist():
    assert os.path.exists("train.csv"), "train.csv bulunamadı"
    assert os.path.exists("test.csv"), "test.csv bulunamadı"


def test_main_imports():
    mod = importlib.import_module("main")
    # Script seviyesinde çalıştırmadan sadece import edilebildiğini doğrula
    assert mod is not None


def test_advanced_pipeline_imports():
    mod = importlib.import_module("advanced_pipeline")
    assert mod is not None
