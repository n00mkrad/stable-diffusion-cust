# ControlNet Preprocessors for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
Moved from https://github.com/comfyanonymous/ComfyUI/pull/13 <br>
Original repo: https://github.com/lllyasviel/ControlNet <br>
List of my comfyUI node repos: https://github.com/Fannovel16/FN16-ComfyUI-nodes
### UPDATED ONE-CLICK DEPENDENCIES INSTALLATION METHOD. CHECK OUT THE INSTALL SECTION

## Change log:
### 2023-04-01
* Renamed MediaPipePreprocessor to MediaPipe-PoseHandPreprocessor to avoid confusion
* Added MediaPipe-FaceMeshPreprocessor for [ControlNet Face Model](https://www.reddit.com/r/StableDiffusion/comments/1281iva/new_controlnet_face_model/)
### 2023-04-02
* Fixed https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/20
* Fixed typo at ##Nodes
### 2023-04-10
* Fixed https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/18, https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/25: https://github.com/Fannovel16/comfy_controlnet_preprocessors/commit/b8a108a0f8ae37b9302b32d7c236cfa3dde97920, https://github.com/Fannovel16/comfy_controlnet_preprocessors/commit/01fbab5cdfc7b013256d4aec4e5ad77edb80a039
### 2023-04-20
* Added HED-v11-Preprocessor, PiDiNet-v11-Preprocessor, Zoe-DepthMapPreprocessor and BAE-NormalMapPreprocessor
* If you installed this repo, please run `install.py` after `git pull` if you want to download all files for new preprocessors for offline run
* Removed the needs of BasicSR (https://github.com/Fannovel16/comfy_controlnet_preprocessors/commit/6b073101f75d6ab1e53c231dab8118990fec96ed) since ComfyUI's custom Python build can't install it.
### 2023-04-22
* Merged HED-v11-Preprocessor, PiDiNet-v11-Preprocessor into HEDPreprocessor and PiDiNetPreprocessor. They now use v1.1 version by default. Set `version` to `v1` to get old results
* Added `safe` options to these two.
* Editing Nodes section
* Updated single-click dependecies installation method. Check out the install section.
* Added Openpose preprocessor v1.1, TilePreprocessor
### 2023-04-26
* Added UniFormer-SemSegPreprocessor (alias of SemSegPreprocessor), OneFormer-COCO-SemSegPreprocessor, OneFormer-ADE20K-SemSegPreprocessor, LineArtPreprocessor, AnimeLineArtPreprocessor
* Fixed https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/37 and https://github.com/Fannovel16/comfy_controlnet_preprocessors/issues/36
* Fixed typos at install.py
* Fixed issues from OneFormer
* Added Manga2Anime-LineArtPreprocessor
### 2023-05-06
* Fixed #34
### 2023-05-09
* Add support for DirectML (replace `map_location=get_torch_device()` with `.to(get_torch_device())`)
* Merge https://github.com/Fannovel16/comfy_controlnet_preprocessors/pull/45 to fix import errors on Linux
### 2023-05-10
* Remove reportlab and svglib which are useless and can't be built on Colab

## Usage
All preprocessor nodes take an image, usually came from LoadImage node and output a map image (aka hint image):
* The input image can have any kind of resolution, not need to be multiple of 64. They will be resized to fit the nearest multiple-of-64 resolution behind the scene.
* The hint image is a black canvas with a/some subject(s) like Openpose stickman(s), depth map, etc

If a preprocessor node doesn't have `version` option, it is unchanged in ControlNet 1.1.

It is recommended to use version `v1.1` of preprocessors if they have `version` option since results from v1.1 preprocessors are better than v1 one and compatibile with both ControlNet 1 and ControlNet 1.1.

If you want to reproduce results from old workflows, set `version` to `v1` if it exists.

## Nodes
### Canny Edge
| Preprocessor Node           | sd-webui-controlnet/other | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| CannyEdgePreprocessor       | canny                     | control_v11p_sd15_canny <br> control_canny <br> t2iadapter_canny | preprocessors/edge_line |

| Source | Input | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_input.png?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro_canny.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/mahiro-out.png?raw=true"> |

### Normal, Coarse and Anime Line Art
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| LineArtPreprocessor         | lineart (or `lineart_coarse` if `coarse` is enabled)  | control_v11p_sd15_lineart                 | preprocessors/edge_line          |
| AnimeLineArtPreprocessor    | lineart_anime                                         | control_v11p_sd15s2_lineart_anime         | preprocessors/edge_line          |
|Manga2Anime-LineArtPreprocessor| lineart_anime                                         | control_v11p_sd15s2_lineart_anime         | preprocessors/edge_line          |

### M-LSD
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| M-LSDPreprocessor           | mlsd                                                  | control_v11p_sd15_mlsd <br> control_mlsd  | preprocessors/edge_line          |

Example images: WIP

### Scribble and Fake Scribble
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| ScribblePreprocessor        | scribble                                              |control_v11p_sd15_scribble <br> control_scribble| preprocessors/edge_line          |
| FakeScribblePreprocessor    | fake_scribble                                         |control_v11p_sd15_scribble <br> control_scribble| preprocessors/edge_line          |

Example images: WIP

### Soft Edge (HED and PiDiNet)
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| HEDPreprocessor             | hed                                                   | control_v11p_sd15_softedge <br> control_hed | preprocessors/edge_line          |
| PiDiNetPreprocessor         | pidinet                                               | control_v11p_sd15_softedge <br> control_scribble <br> t2iadapter_sketch | preprocessors/edge_line          |

#### HED
* THE NEW SOFTEDGE HED IS CALLED HED 1.1 IN THIS REPO. IT IS ENABLED BY DEFAULT AS value `v1.1` in the version field
* v1 uses Saining Xie's official implementation which uses GPL. v1.1 uses lllyasviel's own implementation which doesn't contain GPL contamination.
* v1.1 generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
* You can only use `safe` option if the version is `v1.1`, otherwise, it is ignored.

| Source | Input | Output |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_source.jpg?raw=true">  |  <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_hed.png?raw=true"> | <img width="256" alt="" src="https://github.com/Mikubill/sd-webui-controlnet/blob/main/samples/evt_gen.png?raw=true"> |

#### PiDiNet (WIP)

### Depth Map
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| MiDaS-DepthMapPreprocessor  | (normal) depth                                        | control_v11f1p_sd15_depth <br> control_depth <br> t2iadapter_depth       | preprocessors/normal_depth_map   |
| LeReS-DepthMapPreprocessor  | depth_leres                                           | control_v11f1p_sd15_depth <br> control_depth <br> t2iadapter_depth       | preprocessors/normal_depth_map   |
| Zoe-DepthMapPreprocessor    | depth_zoe                                             | control_v11f1p_sd15_depth <br> control_depth <br> t2iadapter_depth       | preprocessors/normal_depth_map   |

### Openpose
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| OpenposePreprocessor        | openpose (detect_body) <br> openpose_hand (detect_body + detect_hand) <br> openpose_faceonly (detect_face) <br> openpose_full (detect_hand + detect_body + detect_face) | control_v11p_sd15_openpose <br> control_openpose <br> t2iadapter_openpose | preprocessors/pose |

### Normal Map
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| MiDaS-NormalMapPreprocessor | normal_map                                            | control_normal                            | preprocessors/normal_depth_map   |
| BAE-NormalMapPreprocessor   | normal_map                                            | control_v11p_sd15_normalbae               | preprocessors/normal_depth_map   |

* You should use BAE's normal map instead of MiDaS's one because it gives way better results.

### Tile
| Preprocessor Node           | sd-webui-controlnet/other | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| TilePreprocessor            |                           | control_v11u_sd15_tile                    | preprocessors/tile   |

### Semantic Segmantation
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| UniFormer-SemSegPreprocessor / SemSegPreprocessor | segmentation <br> Seg_UFADE20K  | control_v11p_sd15_seg <br> control_seg <br> t2iadapter_seg           | preprocessors/semseg          |
| OneFormer-COCO-SemSegPreprocessor | oneformer_coco                                  | control_v11p_sd15_seg                    | preprocessors/semseg   |
| OneFormer-ADE20K-SemSegPreprocessor | oneformer_ade20k                              | control_v11p_sd15_seg                    | preprocessors/semseg   |

* UniFormer-SemSegPreprocessor is a new alias for SemSegPreprocessor. Any new workflow should use it instead of SemSegPreprocessor to avoid confusion. It is kept for backward compatibility.

### Others
| Preprocessor Node           | sd-webui-controlnet/other                             | Use with ControlNet/T2I-Adapter           | Category                         |
|-----------------------------|-------------------------------------------------------|-------------------------------------------|----------------------------------|
| BinaryPreprocessor          | binary                                                | control_scribble                          | preprocessors/edge_line          |
|MediaPipe-PoseHandPreprocessor| https://natakaro.gumroad.com/l/oprmi                 | https://civitai.com/models/16409          | preprocessors/pose               |
| ColorPreprocessor           | color                                                 | t2iadapter_color                          | preprocessors/color_style        |
|MediaPipe-FaceMeshPreprocessor| mediapipe_face                                       | controlnet_sd21_laion_face_v2             | preprocessors/face_mesh          |

## Install
Firstly, [install comfyui's dependencies](https://github.com/comfyanonymous/ComfyUI#installing) if you didn't.
Then run:
```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors
cd comfy_controlnet_preprocessors
```
Add `--no_download_ckpts` to the command in below methods if you don't want to download any model. <br>
When a preprocessor node runs, if it can't find the models it need, that models will be downloaded automatically.
### New dependencies installation method
Open the terminal then run
```sh
install
```
It will automatically find out what Python's build should be used and use it to run install.py
### Old one
Next, run install.py. It will download all models by default. <br>
Note that you have to check if ComfyUI you are using is portable standalone build or not. If you use the wrong command, requirements won't be installed in the right place. 
For directly-cloned ComfyUI repo, run:
```
python install.py
```
For ComfyUI portable standalone build:
```
/path/to/ComfyUI/python_embeded/python.exe install.py
```

## Apple Silicon
A few preprocessors utilize operators not implemented for Apple Silicon MPS device, yet.
For example, `Zoe-DepthMapPreprocessor` depends on `aten::upsample_bicubic2d.out` operator.
Thus you should enable `$PYTORCH_ENABLE_MPS_FALLBACK`.
This makes sure unimplemented operators are calculated by the CPU.

```sh 
PYTORCH_ENABLE_MPS_FALLBACK=1 python /path/to/ComfyUI/main.py
```

## Model Dependencies 
The total disk's free space needed if all models are downloaded is ~1.58 GB. <br>
All models will be downloaded to `comfy_controlnet_preprocessors/ckpts`
* network-bsds500.pth (hed): 56.1 MB
* res101.pth (leres): 506 MB
* dpt_hybrid-midas-501f0c75.pt (midas): 470 MB
* mlsd_large_512_fp32.pth (mlsd): 6 MB
* body_pose_model.pth (for both openpose v1 and v1.1): 200 MB
* hand_pose_model.pth (for both openpose v1 and v1.1): 141 MB
* facenet.pth (openpose v1.1): 154 MB
* upernet_global_small.pth (uniformer aka SemSeg): 197 MB
* table5_pidinet.pth (for both PiDiNet v1 and v1.1): 2.87 MB
* ControlNetHED.pth (New HED 1.1): 29.4 MB
* scannet.pt (NormalBAE): 291 MB

## Limits
* There may be bugs since I don't have time ~~(lazy)~~ to test
* ~~You must have CUDA device because I just put `.cuda()` everywhere.~~ It is fixed

## Citation
### Original ControlNet repo
    @misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Maneesh Agrawala},
      year={2023},
      eprint={2302.05543},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

[Arxiv Link](https://arxiv.org/abs/2302.05543)

### Mikubill/sd-webui-controlnet
https://github.com/Mikubill/sd-webui-controlnet
### Others:
* https://natakaro.gumroad.com/l/oprmi
