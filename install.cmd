@echo off

echo.
echo Downloading Stable Diffusion 1.4 model file...

curl "https://drive.yerf.org/wl/?id=EBfTrmcCCUAGaQBXVIj5lJmEhjoP1tgl&mode=grid&download=1" -o model.ckpt

SET CONDA_PATH=..\mc\Scripts

echo.
echo Setting up python environment...

call "%CONDA_PATH%\activate.bat"
call conda env create -f environment.yaml
call conda env update --file environment.yaml --prune
call "%CONDA_PATH%\activate.bat" ldo

rem rmdir /s /q src

rmdir /s /q .git
rmdir /s /q src/taming-transformers/.git