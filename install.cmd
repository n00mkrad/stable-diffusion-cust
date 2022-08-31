@echo off

SET CONDA_PATH=..\mc\Scripts

echo.
echo PRINTME Setting up python environment...

call "%CONDA_PATH%\activate.bat"
call conda env create -f environment.yaml
call conda env update --file environment.yaml --prune
call "%CONDA_PATH%\activate.bat" ldo
