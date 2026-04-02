# Honours-Stage-Project-25-26---Interactive Neural Network Dashboard
[![Python CI](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/ci.yml)

A final year project to design an Interactive Dashboard that will serve as a walkthrough in using Feed-Forward Neural Networks to solve a classification problem, the Iris Dataset

# Development Setup

## New Machine

1. Install Python 3.13.
2. Open this project folder in VS Code.
3. Run these commands in PowerShell from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

4. In VS Code, run `Python: Select Interpreter`.
5. Choose `.venv\Scripts\python.exe`.

## Run The App

```powershell
.\.venv\Scripts\python.exe app.py
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
.\.venv\Scripts\python.exe -m pytest -v pytests
```

