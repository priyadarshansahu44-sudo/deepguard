@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run.py
) else if exist "C:\Users\SACHIN\Documents\Codex\2026-10-30\task-compliance-md-use-assignment-requirement\work\.venv\Scripts\python.exe" (
  "C:\Users\SACHIN\Documents\Codex\2026-10-30\task-compliance-md-use-assignment-requirement\work\.venv\Scripts\python.exe" run.py
) else (
  python run.py
)
pause
