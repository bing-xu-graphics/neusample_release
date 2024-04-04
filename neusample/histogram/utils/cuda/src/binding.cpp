#include "sample_texture.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_texture_2d", &sample_texture_2d);
    m.def("sample_multi_texture_2d", &sample_multi_texture_2d);
    m.def("sample_batch_texture_2d", &sample_batch_texture_2d);
    m.def("sample_idx_texture_2d", &sample_idx_texture_2d);
    m.def("fetch_idx_texture_2d", &fetch_idx_texture_2d);
}