# Honours-Stage-Project-25-26---Interactive Neural Network Dashboard

## Quality Control

### Aim 1 - Learning Interface
- [![Objective 1.1 - Learning Interface](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-1-1-learning-interface.yml/badge.svg)](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-1-1-learning-interface.yml)
- [![Objective 1.2 - Adaptive Learning](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-1-2-adaptive-learning.yml/badge.svg)](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-1-2-adaptive-learning.yml)

### Aim 2 - Model Factory Subsystem
- [![Objective 2.1 - Model Fabrication and Testing](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-2-1-model-fabrication-testing.yml/badge.svg)](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-2-1-model-fabrication-testing.yml)
- [![Objective 2.2 - Data Handling](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-2-2-data-handling.yml/badge.svg)](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-2-2-data-handling.yml)

### Aim 3 - Secure and Deployable Application
- [![Objective 3.1 - Secure Code Execution](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-3-1-secure-code-execution.yml/badge.svg)](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-3-1-secure-code-execution.yml)
- [![Objective 3.2 - Deployment and Persistence](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-3-2-deployment-persistence.yml/badge.svg)](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/objective-3-2-deployment-persistence.yml)

### Deployment
[![Deploy To Cloud Run](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/deploy.yml/badge.svg)](https://github.com/HarryDixon-Hall/Honours-Stage-Project-25-26---Interactive-Neural-Network-Dashboard/actions/workflows/deploy.yml)

Configure GitHub branch protection on `main` to require these six objective workflows before merge:

1. Objective 1.1 - Learning Interface
2. Objective 1.2 - Adaptive Learning
3. Objective 2.1 - Model Fabrication and Testing
4. Objective 2.2 - Data Handling
5. Objective 3.1 - Secure Code Execution
6. Objective 3.2 - Deployment and Persistence

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

