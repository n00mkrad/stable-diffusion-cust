@echo off

SET CONDA_PATH=..\mc\Scripts

echo.
echo PRINTME Setting up python environment...

call "%CONDA_PATH%\activate.bat"
call conda env create -f environment.yaml -p "%CONDA_PATH%\envs\ldo"
call conda env update --file environment.yaml --prune -p "%CONDA_PATH%\envs\ldo"
call "%CONDA_PATH%\activate.bat" "%CONDA_PATH%\envs\ldo"