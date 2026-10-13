@echo off
setlocal EnableExtensions
cd /d "%~dp0"
where conda >nul 2>&1
if errorlevel 1 (
  echo ERROR: Conda is not on PATH. Open Anaconda Prompt or Miniconda Prompt and rerun setup_conda.bat.
  exit /b 1
)
conda env list | findstr /I /C:"leukemia-yolo" >nul
if errorlevel 1 (
  echo Creating leukemia-yolo from environment.yml ...
  call conda env create -f environment.yml
  if errorlevel 1 exit /b 1
) else (
  echo Existing leukemia-yolo environment found.
)
call conda activate leukemia-yolo
if errorlevel 1 (
  echo ERROR: Conda activation failed.
  exit /b 1
)
python -c "import sys; assert sys.version_info[:2] == (3,11); import notebook,jupyterlab,ipykernel; print('Python and notebook packages ready:',sys.executable)"
if errorlevel 1 exit /b 1
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo WARNING: nvidia-smi was not found. Check the NVIDIA driver before selecting a CUDA PyTorch wheel.
) else (
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)
python -c "import torch,torchvision; assert torch.version.cuda is not None and torch.cuda.is_available(); print('CUDA PyTorch:',torch.__version__,'torchvision:',torchvision.__version__,'GPU:',torch.cuda.get_device_name(0))"
if errorlevel 1 goto :manual_torch
python -c "import importlib.metadata as m; print('torch=='+m.version('torch')); print('torchvision=='+m.version('torchvision'))" > .torch-constraints.txt
if errorlevel 1 exit /b 1
echo Installing remaining packages while pinning the working torch and torchvision builds ...
python -m pip install --constraint .torch-constraints.txt -r requirements-no-torch.txt
if errorlevel 1 (
  echo ERROR: Dependency resolution failed. Existing PyTorch was not intentionally replaced. Check the conflict before retrying.
  exit /b 1
)
python -m pip install --no-deps -e .
if errorlevel 1 exit /b 1
python -m ipykernel install --user --name leukemia-yolo --display-name "Python (leukemia-yolo)"
if errorlevel 1 exit /b 1
python -c "import torch,torchvision,ultralytics,cv2,PIL,numpy,yaml,sklearn,pytorch_grad_cam,src; assert torch.version.cuda is not None and torch.cuda.is_available(); print('Imports OK. CUDA GPU:',torch.cuda.get_device_name(0),'free/total GiB:',*[round(v/2**30,2) for v in torch.cuda.mem_get_info(0)])"
if errorlevel 1 exit /b 1
echo.
echo READY: Copy config.example.yaml to config.local.yaml, set dataset_root, then run launch_jupyter.bat.
echo Select kernel Python (leukemia-yolo) and open notebooks/00_Environment_Check.ipynb first.
exit /b 0
:manual_torch
echo.
echo STOP: A working CUDA-enabled PyTorch and torchvision pair is not installed or CUDA is unavailable.
echo 1. Run nvidia-smi and confirm the NVIDIA driver and GPU are visible.
echo 2. Visit https://pytorch.org/get-started/locally/
echo 3. Select Windows / Pip / Python / CUDA version supported by your driver.
echo 4. After this script exits, run: conda activate leukemia-yolo
echo 5. Run EXACTLY the torch and torchvision install command shown by the selector in that environment.
echo 6. Verify: python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
echo 7. Rerun setup_conda.bat. No CPU-only torch installation was attempted.
exit /b 2
