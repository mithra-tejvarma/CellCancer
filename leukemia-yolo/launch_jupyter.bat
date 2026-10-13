@echo off
setlocal EnableExtensions
cd /d "%~dp0"
where conda >nul 2>&1
if errorlevel 1 (
  echo ERROR: Open Anaconda Prompt or Miniconda Prompt first.
  exit /b 1
)
call conda activate leukemia-yolo
if errorlevel 1 exit /b 1
python -c "import sys; assert 'leukemia-yolo' in sys.prefix.lower(), sys.executable"
if errorlevel 1 (
  echo ERROR: The Python kernel is not in leukemia-yolo.
  exit /b 1
)
jupyter lab notebooks
