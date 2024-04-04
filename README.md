# neusample_release
Code and resources for SIGGRAPH 2023  paper NeuSample: Importance Sampling for Neural Materials 


# 6D BRDF data 
- https://drive.google.com/drive/folders/10vPIMHrnFlLMj7uPatTWRKdARu4_JlhD?usp=sharing 
- 40+ different materials for public use including anisotropic and multi-layer; not all are used in the original paper. Credit to Fujun Luan (model training) and Alexandr Kuznetsov, Krishna Mullia (NeuMIP implementation).
- this is in the format of training weights .ckpt, which can be used by Adobe version of simplified [NeuMIP](https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP/) implementation (credit to Krishna Mullia and Alexandr Kuznetsov).
- 6D data: 2D for incoming direction; 2D for outgoing direction; 2D for surface uv. Please refer to NeuMIP and NeuSample papers for more details.
