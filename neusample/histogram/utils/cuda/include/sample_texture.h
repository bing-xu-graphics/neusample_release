#pragma once
#include <torch/extension.h>
#include <utility>

void sample_texture_2d(
    at::Tensor cdf_y,
    at::Tensor cdf_x,
    at::Tensor samples,
    at::Tensor wis,
    at::Tensor pdfs
);

void sample_multi_texture_2d(
    at::Tensor cdf_z,
    at::Tensor cdf_y,
    at::Tensor cdf_x,
    at::Tensor samples,
    at::Tensor wis,
    at::Tensor pdfs
);

void sample_batch_texture_2d(
    at::Tensor cdf_y,
    at::Tensor cdf_x,
    at::Tensor samples,
    at::Tensor wis,
    at::Tensor pdfs
);

void sample_idx_texture_2d (
    at::Tensor idx,
    at::Tensor phi,
    at::Tensor cdf_z,
    at::Tensor cdf_y,
    at::Tensor cdf_x,
    at::Tensor samples,
    at::Tensor wis,
    at::Tensor pdfs
);

void fetch_idx_texture_2d (
    at::Tensor idx,
    at::Tensor phi,
    at::Tensor pdf_z,
    at::Tensor pdf_xy,
    at::Tensor wis,
    at::Tensor pdfs
);