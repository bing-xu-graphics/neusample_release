#include "sample_texture.h"
#include <utility>

void sample_texture_2d_wrapper(
    const int batch_size,
    const float* cdf_y,
    const float* cdf_x,
    const float* samples,
    float* wis,
    float* pdfs
);

void sample_multi_texture_2d_wrapper(
    const int batch_size,
    const float* cdf_z,
    const float* cdf_y,
    const float* cdf_x,
    const float* samples,
    float* wis,
    float* pdfs
);

void sample_batch_texture_2d_wrapper(
    const int batch_size,
    const float* cdf_y,
    const float* cdf_x,
    const float* samples,
    float* wis,
    float* pdfs
);

void sample_idx_texture_2d_wrapper (
    const int batch_size,
    const int* idx,
    const float* phi,
    const float* cdf_z,
    const float* cdf_y,
    const float* cdf_x,
    const float* samples,
    float* wis,
    float* pdfs
);

void fetch_idx_texture_2d_wrapper (
    const int batch_size,
    const int* idx,
    const float* phi,
    const float* pdf_z,
    const float* pdf_xy,
    const float* wis,
    float* pdfs
);

void sample_texture_2d (
    at::Tensor cdf_y,
    at::Tensor cdf_x,
    at::Tensor samples,
    at::Tensor wis,
    at::Tensor pdfs
) {
    auto batch_size = samples.size(0);
    sample_texture_2d_wrapper(
        batch_size,
        cdf_y.data_ptr<float>(),
        cdf_x.data_ptr<float>(),
        samples.data_ptr<float>(),
        wis.data_ptr<float>(),
        pdfs.data_ptr<float>()
    );
}

void sample_multi_texture_2d (
    at::Tensor cdf_z,
    at::Tensor cdf_y,
    at::Tensor cdf_x,
    at::Tensor samples,
    at::Tensor wis,
    at::Tensor pdfs
) {
    auto batch_size = samples.size(0);
    sample_multi_texture_2d_wrapper(
        batch_size,
        cdf_z.data_ptr<float>(),
        cdf_y.data_ptr<float>(),
        cdf_x.data_ptr<float>(),
        samples.data_ptr<float>(),
        wis.data_ptr<float>(),
        pdfs.data_ptr<float>()
    );
}

void sample_batch_texture_2d(
    at::Tensor cdf_y,
    at::Tensor cdf_x,
    at::Tensor samples,
    at::Tensor wis,
    at::Tensor pdfs
) {
    auto batch_size = samples.size(0);
    sample_batch_texture_2d_wrapper(
        batch_size,
        cdf_y.data_ptr<float>(),
        cdf_x.data_ptr<float>(),
        samples.data_ptr<float>(),
        wis.data_ptr<float>(),
        pdfs.data_ptr<float>()
    );
}

void sample_idx_texture_2d (
    at::Tensor idx,
    at::Tensor phi,
    at::Tensor cdf_z,
    at::Tensor cdf_y,
    at::Tensor cdf_x,
    at::Tensor samples,
    at::Tensor wis,
    at::Tensor pdfs
) {
    auto batch_size = samples.size(0);
    sample_idx_texture_2d_wrapper(
        batch_size,
        idx.data_ptr<int>(),
        phi.data_ptr<float>(),
        cdf_z.data_ptr<float>(),
        cdf_y.data_ptr<float>(),
        cdf_x.data_ptr<float>(),
        samples.data_ptr<float>(),
        wis.data_ptr<float>(),
        pdfs.data_ptr<float>()
    );
}

void fetch_idx_texture_2d (
    at::Tensor idx,
    at::Tensor phi,
    at::Tensor pdf_z,
    at::Tensor pdf_xy,
    at::Tensor wis,
    at::Tensor pdfs
) {
    auto batch_size = wis.size(0);
    fetch_idx_texture_2d_wrapper (
        batch_size,
        idx.data_ptr<int>(),
        phi.data_ptr<float>(),
        pdf_z.data_ptr<float>(),
        pdf_xy.data_ptr<float>(),
        wis.data_ptr<float>(),
        pdfs.data_ptr<float>()
    );
}