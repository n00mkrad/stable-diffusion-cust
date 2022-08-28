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

rem call "%CONDA_PATH%\activate.bat"
call "%CONDA_PATH%\activate.bat" ldo
rem python "%CD%"\scripts\relauncher.py

:PROMPT
set SETUPTOOLS_USE_DISTUTILS=stdlib
IF EXIST "model.ckpt" (
  python scripts/dream.py
) ELSE (
  ECHO model.ckpt does not exist!
)

pause