@echo off

SET CONDA_PATH=..\mc\Scripts

call "%CONDA_PATH%\activate.bat"
call "%CONDA_PATH%\activate.bat" "%CONDA_PATH%\envs\ldo"

pip install realesrgan