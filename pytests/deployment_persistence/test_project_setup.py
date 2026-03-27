from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pytest_ini_targets_pytests_directory():
    pytest_ini = (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8")

    assert "testpaths = pytests" in pytest_ini
    assert "python_functions = test_*" in pytest_ini


def test_requirements_dev_includes_pytest_dependencies():
    requirements_dev = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")

    assert "pytest==" in requirements_dev
    assert "pytest-cov==" in requirements_dev


def test_pyinstaller_spec_targets_app_entrypoint():
    spec_file = (REPO_ROOT / "app.spec").read_text(encoding="utf-8")

    assert "['app.py']" in spec_file
    assert "name='app'" in spec_file


def test_progress_persistence_module_exists_for_future_expansion():
    database_module = REPO_ROOT / "distribution" / "database.py"

    assert database_module.exists()