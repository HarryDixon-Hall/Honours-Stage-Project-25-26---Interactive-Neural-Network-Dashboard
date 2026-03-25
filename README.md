# Honours-Stage-Project-25-26---Interactive Neural Network Dashboard
A final year project to design an Interactive Dashboard that will serve as a walkthrough in using Feed-Forward Neural Networks to solve a classification problem, the Iris Dataset

# Important

## Python Environment

This workspace uses a selected VS Code interpreter instead of a hard-coded interpreter path in workspace settings.

Recommended interpreter location on each Windows machine:

- `%LOCALAPPDATA%\interactive-dashboard\.venv\Scripts\python.exe`

- VS Code debug launch: [.vscode/launch.json](.vscode/launch.json)

Using an external environment avoids package corruption caused by OneDrive syncing local virtual environment files.
Using the selected interpreter instead of a saved absolute path makes F5 more robust across different machines.

## Setup

Create the environment and install the pinned dependency set from [requirements.txt](requirements.txt):

```powershell
$VenvPath = Join-Path $env:LOCALAPPDATA "interactive-dashboard/.venv"
py -3.13 -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r requirements.txt
```

## Running The App

Run the dashboard from the project root:

```powershell
$VenvPath = Join-Path $env:LOCALAPPDATA "interactive-dashboard/.venv"
& "$VenvPath\Scripts\python.exe" app.py
```

The Dash development server will start on `http://127.0.0.1:8050/`.

## Running In VS Code

- Run `Python: Select Interpreter`
- Choose `%LOCALAPPDATA%\interactive-dashboard\.venv\Scripts\python.exe`
- Press `F5` and choose `Run Dash App`
- No manual activation step is required

## Multi-Machine Workflow

1. Open project on any Windows machine.
2. Run the Setup commands once on that machine.
3. Run `Python: Select Interpreter` and select `%LOCALAPPDATA%\interactive-dashboard\.venv\Scripts\python.exe`.
4. Press F5 and launch `Run Dash App`.

Because `%LOCALAPPDATA%` is local to each PC and not synced by OneDrive, this avoids cross-machine virtual environment breakage.
Because the interpreter is selected per machine instead of stored as a fixed path in the repo settings, F5 follows the correct local environment more reliably.

## If running the application for the first time on a machine

1. run 
$VenvPath = Join-Path $env:LOCALAPPDATA "interactive-dashboard/.venv"
py -3.13 -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r .\requirements.txt

## Testing In VS Code

Use the Python extension's Testing view for discovery, running, and debugging tests.

Install the development test dependencies:

```powershell
$VenvPath = Join-Path $env:LOCALAPPDATA "interactive-dashboard/.venv"
& "$VenvPath\Scripts\python.exe" -m pip install -r .\requirements-dev.txt
```

### Current workspace setup

- `pytest` is enabled by default in VS Code.
- Test discovery folder for `pytest`: `pytests/`
- Test file pattern: `test_*.py`
- `pytest` project config: [pytest.ini](pytest.ini)
- Test debug launch config: [.vscode/launch.json](.vscode/launch.json)
- Starter examples:
	- [pytests/test_dataload_pytest.py](pytests/test_dataload_pytest.py)

### How to use the Testing view

1. Open the project in VS Code.
2. Run `Python: Select Interpreter` and choose `%LOCALAPPDATA%\interactive-dashboard\.venv\Scripts\python.exe`.
3. Open the Testing view from the beaker icon in the Activity Bar.
4. Press Refresh Tests if discovery does not run automatically.
5. Use the run or debug icons beside a test, file, or folder.
6. Expand the test tree to see each individual test with its pass or fail status.

### Running pytest from the terminal

Run all pytest tests:

```powershell
$VenvPath = Join-Path $env:LOCALAPPDATA "interactive-dashboard/.venv"
& "$VenvPath\Scripts\python.exe" -m pytest -v .\pytests
```

Run one pytest file:

```powershell
$VenvPath = Join-Path $env:LOCALAPPDATA "interactive-dashboard/.venv"
& "$VenvPath\Scripts\python.exe" -m pytest -v .\pytests\test_dataload_pytest.py
```

Run one specific pytest test function:

```powershell
$VenvPath = Join-Path $env:LOCALAPPDATA "interactive-dashboard/.venv"
& "$VenvPath\Scripts\python.exe" -m pytest -v .\pytests\test_dataload_pytest.py -k "test_load_dataset_iris_returns_consistent_shapes"
```

### Seeing which tests passed

In VS Code:

1. Open the Testing view.
2. Expand the `pytests` tree.
3. Each test will show its own status icon.
4. Select a test to inspect its latest run result.

In the terminal:

1. Use `-v` with pytest to show each discovered test by name.
2. Use `-vv` for even more detail if needed.

Example verbose output:

```text
pytests/test_dataload_pytest.py::test_get_dataset_stats_returns_expected_counts PASSED
pytests/test_dataload_pytest.py::test_load_dataset_iris_returns_consistent_shapes PASSED
```

### Adding more pytest tests

1. Create a new file in `pytests/` named like `test_models.py` or `test_dataload.py`.
2. Write test functions whose names start with `test_`.
3. Import the function or class you want to test from the project module.
4. Refresh the Testing view if the test does not appear automatically.
5. Run the file, a single test, or the full suite from the UI or terminal.

Example:

```python
from dataload import load_dataset


def test_load_dataset_rejects_unknown_name():
	try:
		load_dataset("unknown")
	except ValueError as exc:
		assert "Unknown dataset" in str(exc)
	else:
		raise AssertionError("Expected ValueError")
```

### Debugging tests in VS Code

1. Open the Testing view.
2. Click the debug icon beside a test, file, or folder.
3. VS Code will use the `Python: Debug Tests` configuration from [.vscode/launch.json](.vscode/launch.json).
4. Set breakpoints inside your application code or test file before debugging.

