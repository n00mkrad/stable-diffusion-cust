@echo off

SET CONDA_ROOT_PATH=..\mb
SET CONDA_SCRIPTS_PATH=..\mb\Scripts

SET PATH=..\mb\condabin;%PATH%
SET PATH=..\mb\Library\bin;%PATH%
SET PATH=..\mb\Scripts;%PATH%
SET PATH=..\mb\;%PATH%

echo.
echo PRINTME Setting up python environment...

_conda env create -f environment.yaml -p "%CONDA_ROOT_PATH%\envs\ldo"
_conda env update --file environment.yaml --prune -p "%CONDA_ROOT_PATH%\envs\ldo"
call "%CONDA_SCRIPTS_PATH%\activate.bat" "%CONDA_ROOT_PATH%\envs\ldo"