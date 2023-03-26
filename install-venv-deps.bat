
echo Will now install: torch==1.11.0+cu113 torchvision==0.12.0+cu113
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

echo Will now install: albumentations==0.4.3
python -m pip install albumentations==0.4.3

echo Will now install: opencv-python==4.5.5.64
python -m pip install opencv-python==4.5.5.64

echo Will now install: pudb==2019.2
python -m pip install pudb==2019.2

echo Will now install: imageio==2.9.0
python -m pip install imageio==2.9.0

echo Will now install: pytorch-lightning==1.6.5
python -m pip install pytorch-lightning==1.6.5

echo Will now install: omegaconf==2.1.1
python -m pip install omegaconf==2.1.1

echo Will now install: test-tube>=0.7.5
python -m pip install test-tube>=0.7.5

echo Will now install: pillow==9.2.0
python -m pip install pillow==9.2.0

echo Will now install: einops==0.6.0
python -m pip install einops==0.6.0

echo Will now install: torch-fidelity==0.3.0
python -m pip install torch-fidelity==0.3.0

echo Will now install: transformers~=4.26
python -m pip install transformers~=4.26

echo Will now install: torchmetrics==0.6.0
python -m pip install torchmetrics==0.6.0

echo Will now install: realesrgan==0.3.0
python -m pip install realesrgan==0.3.0

echo Will now install: picklescan==0.0.5
python -m pip install picklescan==0.0.5

echo Will now install: safetensors~=0.3.0
python -m pip install safetensors~=0.3.0

echo Will now install: diffusers
python -m pip install git+https://github.com/Stax124/diffusers@5d41b5383c6a0421a24b17f4520d71a1834c1dac

echo Will now install: clip
python -m pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

echo Will now install: taming-transformers
python -m pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

echo Will now install: k_diffusion
python -m pip install git+https://github.com/Birch-san/k-diffusion.git@mps#egg=k_diffusion

echo Will now install: gfpgan
python -m pip install gfpgan==1.3.8

echo Will now install: clipseg
python -m pip install -e git+https://github.com/invoke-ai/clipseg.git@models-rename#egg=clipseg

echo Will now install: local invoke repo
python -m pip install -e ./repo/invoke

echo Will now install: numpy==1.23.4
python -m pip install numpy==1.23.4