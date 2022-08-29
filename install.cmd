@echo off

set paths=..\mc\Scripts

for %%a in (%paths%) do ( 
 if EXIST "%%a\activate.bat" (
    SET CONDA_PATH=%%a
 )
)

IF "%CONDA_PATH%"=="" (
  echo anaconda3/miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html
  exit /b 1 
) else (
  echo anaconda3/miniconda3 detected in %CONDA_PATH%
)

call "%CONDA_PATH%\activate.bat"
call conda env create -f environment.yaml
call conda env update --file environment.yaml --prune
call "%CONDA_PATH%\activate.bat" ldo
rem python "%CD%"\scripts\relauncher.py

conda deactivate
conda deactivate

curl "https://drive.yerf.org/wl/?id=EBfTrmcCCUAGaQBXVIj5lJmEhjoP1tgl&mode=grid&download=1" -o model.ckpt