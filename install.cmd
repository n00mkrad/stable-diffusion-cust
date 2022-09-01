@echo off

SET CONDA_ROOT_PATH=..\mc
SET CONDA_SCRIPTS_PATH=..\mc\Scripts

SET PATH=..\mc\;%PATH%
SET PATH=..\mc\Scripts;%PATH%
SET PATH=..\mc\Library\bin;%PATH%

echo.
echo PRINTME Setting up python environment...

call "%CONDA_SCRIPTS_PATH%\activate.bat"
call conda env create -f environment.yaml -p "%CONDA_ROOT_PATH%\envs\ldo"
call conda env update --file environment.yaml --prune -p "%CONDA_ROOT_PATH%\envs\ldo"
call "%CONDA_SCRIPTS_PATH%\activate.bat" "%CONDA_ROOT_PATH%\envs\ldo"