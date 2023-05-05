@echo off

echo --- Will now install: torch==1.11.0+cu113 torchvision==0.12.0+cu113 ---
python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

echo --- Will now install: albumentations ---
python -m pip install albumentations==0.4.3

echo --- Will now install: opencv-python ---
python -m pip install opencv-python==4.5.5.64

echo --- Will now install: pudb ---
python -m pip install pudb==2019.2

echo --- Will now install: imageio ---
python -m pip install imageio==2.9.0

echo --- Will now install: pytorch-lightning ---
python -m pip install pytorch-lightning==1.6.5

echo --- Will now install: omegaconf ---
python -m pip install omegaconf==2.1.1

echo --- Will now install: test-tube ---
python -m pip install test-tube>=0.7.5

echo --- Will now install: pillow ---
python -m pip install pillow==9.2.0

echo --- Will now install: einops ---
python -m pip install einops==0.6.0

echo --- Will now install: torch-fidelity ---
python -m pip install torch-fidelity==0.3.0

echo --- Will now install: transformers ---
python -m pip install transformers~=4.26

echo --- Will now install: torchmetrics ---
python -m pip install torchmetrics==0.6.0

echo --- Will now install: realesrgan ---
python -m pip install realesrgan==0.3.0

echo --- Will now install: picklescan ---
python -m pip install picklescan==0.0.5

echo --- Will now install: safetensors ---
python -m pip install safetensors~=0.3.0

echo --- Will now install: diffusers ---
python -m pip install git+https://github.com/huggingface/diffusers@6ba0efb9a188b08f5b46565a87c0b3da7ff46af4

echo --- Will now install: clip ---
python -m pip install git+https://github.com/openai/CLIP.git@main#egg=clip

echo --- Will now install: taming-transformers ---
python -m pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

echo --- Will now install: k_diffusion ---
python -m pip install git+https://github.com/Birch-san/k-diffusion.git@363386981fee88620709cf8f6f2eea167bd6cd74#egg=k_diffusion

echo --- Will now install: gfpgan ---
python -m pip install gfpgan==1.3.8

echo --- Will now install: clipseg ---
python -m pip install git+https://github.com/invoke-ai/clipseg.git@1f754751c85d7d4255fa681f4491ff5711c1c288#egg=clipseg

echo --- Will now install: local invoke repo ---
python -m pip install ./repo/invoke

echo --- Will now install: numpy ---
python -m pip install numpy==1.23.4