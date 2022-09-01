@echo off

SET CONDA_ROOT_PATH=..\mc
SET CONDA_SCRIPTS_PATH=..\mc\Scripts

SET PATH=..\mc\;%PATH%
SET PATH=..\mc\Scripts;%PATH%
SET PATH=..\mc\Library\bin;%PATH%


call "%CONDA_SCRIPTS_PATH%\activate.bat"
call "%CONDA_SCRIPTS_PATH%\activate.bat" "%CONDA_ROOT_PATH%\envs\ldo"

pip install realesrgan