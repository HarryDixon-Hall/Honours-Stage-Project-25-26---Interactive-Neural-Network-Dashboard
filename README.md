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

Or press `F5` in VS Code and choose `Run Dash App`.

The app starts at `http://127.0.0.1:8050/`.

## Deployment

The recommended production target for this project is a Linux container deployment.

- Container build and publish are defined in `.github/workflows/deploy.yml`.
- The recommended Google Cloud Run architecture, identity model, and setup steps are documented in `DEPLOYMENT.md`.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v pytests
```

