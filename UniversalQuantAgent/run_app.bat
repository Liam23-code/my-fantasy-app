@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Missing .venv. Run: python -m venv .venv
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m streamlit run "app\app.py"
endlocal
