# neusample_release
Code and resources for SIGGRAPH 2023 paper [NeuSample: Importance Sampling for Neural Materials](https://cseweb.ucsd.edu/~viscomp/projects/neusample/) 


# 6D Spatially-varying BRDF data 
- https://drive.google.com/drive/folders/10vPIMHrnFlLMj7uPatTWRKdARu4_JlhD?usp=sharing 
- 40+ different materials for public use including anisotropic and multi-layer; not all are used in the original paper. Credit to Fujun Luan (model training) and Alexandr Kuznetsov, Krishna Mullia (NeuMIP implementation).
- this is in the format of training weights .ckpt, which can be used by Adobe version of simplified [NeuMIP](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/) implementation (credit to Krishna Mullia and Alexandr Kuznetsov).
- 6D data: 2D for incoming direction; 2D for outgoing direction; 2D for surface uv. Please refer to NeuMIP (original version) and NeuSample (simplified version without mipmap) papers for more details.


# Training script examples for various sampling methods

- baselines
```
  neusample\scripts\train_xs0000_02_baseline.py
  
  neusample\scripts\train_xs0000_05_xie.py
```
- analytical method:
```
  neusample\scripts\train_xs0000_00_analytical.py // or neusample\scripts\train_xs0027_00_analytical.py for two material examples.
```
- normalizing flow:
```
  neusample\scripts\train_xs0000_04_nsf_prior.py
```
- histogram mixture:
```
  neusample\histogram\train_histogram.py
```
  
# Model inference and render using Mitsuba
```
  neusample\histogram\eval_mitsuba.py
  
  neusample\histogram\eval_tiled.py (tiled version)
```

# Utilities for visualization
```
  neusample/scripts/vis_helper.py
```
  
# Please cite our paper if you don't buy us ice cream
```
@inproceedings{xu2023neusample,
  title={NeuSample: Importance Sampling for Neural Materials},
  author={Xu, Bing and Wu, Liwen and Hasan, Milos and Luan, Fujun and Georgiev, Iliyan and Xu, Zexiang and Ramamoorthi, Ravi},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  pages={1--10},
  year={2023}
}
```

Reference:
We built upon https://github.com/VincentStimper/normalizing-flows. Credit goes to them.

Please let us know if you have any questions! 

Bing Xu at b4xu@ucsd.edu
