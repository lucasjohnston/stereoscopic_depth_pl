# stereoscopic_depth_pl

The aim of this project is to develop a depth map based on a conventional stereoscopic 3D video, so that depth maps of non-copyrighted 3D videos can be used as training data for depth mapping models, but at much higher quality than most existing training datasets.

Commercial 3D videos aren't intended to depict accuracy though, they often maximise the "flying out of the screen" effect, which can lead to exaggerated depth maps, so I'm experimenting with MiDaS to deliver a more accurate result.

## Install

```
conda create -n stereoscopic_depth_env python=3.8
conda activate stereoscopic_depth_env
conda install pytorch=1.13.1 pillow fsspec urllib3
pip install -r requirements.txt
```

## MiDaS model

You must choose a MiDaS model to use, and download it to `midas.pt`.
I'm currently testing with `dpt_swin2_large_384` and MiDaS 3.1.
