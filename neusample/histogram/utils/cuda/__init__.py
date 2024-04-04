from torch.utils.cpp_extension import load
from pathlib import Path

_ext_src_root = Path(__file__).parent / 'src'
_ext_include = Path(__file__).parent / 'include'
exts = ['.cpp', '.cu']
_ext_src_files = [
    str(f) for f in _ext_src_root.iterdir() 
    if any([f.name.endswith(ext) for ext in exts])
]

_ext = load(name='sample_texture_ext', 
            sources=_ext_src_files, 
            extra_include_paths=[str(_ext_include)])

sample_texture_2d = _ext.sample_texture_2d
sample_multi_texture_2d = _ext.sample_multi_texture_2d
sample_batch_texture_2d = _ext.sample_batch_texture_2d
sample_idx_texture_2d = _ext.sample_idx_texture_2d
fetch_idx_texture_2d = _ext.fetch_idx_texture_2d