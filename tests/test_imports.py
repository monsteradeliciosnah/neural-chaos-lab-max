import importlib


def test_import_package():
    m = importlib.import_module("neural_chaos_lab_max")
    assert m is not None


def test_import_submodules_smoke():
    for sub in ("systems", "service", "cli", "ui"):
        try:
            importlib.import_module(f"neural_chaos_lab_max.{sub}")
        except ModuleNotFoundError:
            pass
