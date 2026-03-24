# Honours-Stage-Project-25-26---Interactive Neural Network Dashboard
A final year project to design an Interactive Dashboard that will serve as a walkthrough in using Feed-Forward Neural Networks to solve a classification problem, the Iris Dataset

# Important

## Python Environment

This workspace is configured to use an external virtual environment instead of the in-project `.venv` directory.

- Interpreter: `C:/venvs/interactive-dashboard/Scripts/python.exe`
- VS Code workspace setting: [.vscode/settings.json](.vscode/settings.json)
- VS Code debug launch: [.vscode/launch.json](.vscode/launch.json)

Using an external environment avoids package corruption caused by OneDrive syncing local virtual environment files.

## Setup

Create the environment and install the pinned dependency set from [requirements.txt](requirements.txt):

```powershell
py -3.13 -m venv C:\venvs\interactive-dashboard
& "C:\venvs\interactive-dashboard\Scripts\python.exe" -m pip install --upgrade pip
& "C:\venvs\interactive-dashboard\Scripts\python.exe" -m pip install -r requirements.txt
```

## Running The App

Run the dashboard from the project root:

```powershell
& "C:\venvs\interactive-dashboard\Scripts\python.exe" app.py
```

The Dash development server will start on `http://127.0.0.1:8050/`.

## Running In VS Code

- Use the selected workspace interpreter from [.vscode/settings.json](.vscode/settings.json)
- Press `F5` and choose `Run Dash App`
- No manual activation step is required


