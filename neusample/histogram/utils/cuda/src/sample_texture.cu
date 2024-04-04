#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"
#define NUM_THREADS 1024
#define NUM_BLOCKS 256
#define RES 64
#define RES2 32
#define LOBE 10
#define ANGLE 100

__global__ void sample_texture_2d_kernel(
    const int batch_size,
    const float *__restrict__ cdf_y,
    const float *__restrict__ cdf_x,
    const float *__restrict__ samples,
    float *__restrict__ wis,
    float *__restrict__ pdfs
) {
    int batch_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    samples += batch_idx*3;
    wis += batch_idx*2;
    pdfs += batch_idx;
    
    // sample us
    float u = fmaxf(samples[1],0.00001);
    int left = 0;
    int right = RES-1;
    int mid = (left+right)/2;
    float mid_v = cdf_y[mid];
    float left_v = cdf_y[left];
    float right_v = cdf_y[right];
    // find i cdf[i-1] < u <= cdf[i]
    while (left != mid) { 
         
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_y[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    float pdf_y = cdf_y[mid];
    float u_out = float(mid);
    if (mid>0) {
        float p0 = cdf_y[mid-1];
        pdf_y -= p0;
        u_out += fminf((u-p0)/pdf_y,0.9999f);
    } else {
        u_out += fminf(u/pdf_y,0.9999f);
    }

    wis[1] = (u_out/RES)*2-1;
    pdfs[0] = pdf_y;

    // sample x
    cdf_x += mid*RES;

    u = fmaxf(samples[0],0.00001);
    left = 0;
    right = RES-1;
    mid = (left+right)/2;
    mid_v = cdf_x[mid];
    left_v = cdf_x[left];
    right_v = cdf_x[right];

    while (left != mid) { 
         
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_x[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    pdf_y = cdf_x[mid];
    u_out = float(mid);
    if (mid>0) {
        float p0 = cdf_x[mid-1];
        pdf_y -= p0;
        u_out += fminf((u-p0)/pdf_y,0.9999f);
    } else {
        u_out += fminf(u/pdf_y,0.9999f);
    }
    
    wis[0] = (u_out/RES)*2-1;
    pdfs[0] *= pdf_y;
}

__global__ void sample_multi_texture_2d_kernel(
    const int batch_size,
    const float *__restrict__ cdf_z,
    const float *__restrict__ cdf_y,
    const float *__restrict__ cdf_x,
    const float *__restrict__ samples,
    float *__restrict__ wis,
    float *__restrict__ pdfs
) {
    int batch_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    samples += batch_idx*3;
    wis += batch_idx*2;
    pdfs += batch_idx;
    cdf_z += batch_idx*LOBE;

    // sample lobe
    float u = fmaxf(samples[2],0.00001);
    int left = 0;
    int right = LOBE-1;
    int mid = (left+right)/2;
    float mid_v = cdf_z[mid];
    float left_v = cdf_z[left];
    float right_v = cdf_z[right];

    // if zero lobe, use diffuse sampling
    if (u>right_v) {
        float sin_theta = sqrtf(samples[0]);
        float phi = 2*3.141592653589793f*samples[1];

        wis[0] = sin_theta*cosf(phi);
        wis[1] = sin_theta*sinf(phi);
        
        // != -1 indicates diffuse sampling
        pdfs[0] = 1.0f/3.141592653589793f;
        return;
    } else {
        // incidates pdf still needs to be sampled
        pdfs[0] = -1;
    }

    while (left != mid) {
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_z[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    cdf_y += mid*RES;
    cdf_x += mid*RES*RES;


    // sample us
    u = fmaxf(samples[1],0.00001);
    left = 0;
    right = RES-1;
    mid = (left+right)/2;
    mid_v = cdf_y[mid];
    left_v = cdf_y[left];
    right_v = cdf_y[right];
    // find i cdf[i-1] < u <= cdf[i]
    while (left != mid) { 
         
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_y[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    float pdf_y = cdf_y[mid];
    float u_out = float(mid);
    if (mid>0) {
        float p0 = cdf_y[mid-1];
        pdf_y -= p0;
        u_out += fminf((u-p0)/pdf_y,0.9999f);
    } else {
        u_out += fminf(u/pdf_y,0.9999f);
    }

    wis[1] = (u_out/RES)*2-1;

    // sample x
    cdf_x += mid*RES;

    u = fmaxf(samples[0],0.00001);
    left = 0;
    right = RES-1;
    mid = (left+right)/2;
    mid_v = cdf_x[mid];
    left_v = cdf_x[left];
    right_v = cdf_x[right];

    while (left != mid) { 
         
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_x[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    pdf_y = cdf_x[mid];
    u_out = float(mid);
    if (mid>0) {
        float p0 = cdf_x[mid-1];
        pdf_y -= p0;
        u_out += fminf((u-p0)/pdf_y,0.9999f);
    } else {
        u_out += fminf(u/pdf_y,0.9999f);
    }
    
    wis[0] = (u_out/RES)*2-1;
}

__global__ void sample_batch_texture_2d_kernel(
    const int batch_size,
    const float *__restrict__ cdf_y,
    const float *__restrict__ cdf_x,
    const float *__restrict__ samples,
    float *__restrict__ wis,
    float *__restrict__ pdfs
) {
    int batch_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    samples += batch_idx*3;
    wis += batch_idx*2;
    pdfs += batch_idx;
    cdf_y += batch_idx*RES2;
    cdf_x += batch_idx*RES2*RES2;


    // sample us
    float u = fmaxf(samples[1],0.00001);
    int left = 0;
    int right = RES2-1;
    int mid = (left+right)/2;
    float mid_v = cdf_y[mid];
    float left_v = cdf_y[left];
    float right_v = cdf_y[right];

    // if zero lobe switch to diffuse sampling
    if (u > right_v) {
        float sin_theta = sqrtf(samples[0]);
        float phi = 2*3.141592653589793f*samples[1];

        wis[0] = sin_theta*cosf(phi);
        wis[1] = sin_theta*sinf(phi);
        
        // != -1 indicates diffuse sampling
        pdfs[0] = 1.0f/3.141592653589793f;
        return;
    }


    // find i cdf[i-1] < u <= cdf[i]
    while (left != mid) { 
         
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_y[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    float pdf_y = cdf_y[mid];
    float u_out = float(mid);
    if (mid>0) {
        float p0 = cdf_y[mid-1];
        pdf_y -= p0;
        u_out += fminf((u-p0)/pdf_y,0.9999f);
    } else {
        u_out += fminf(u/pdf_y,0.9999f);
    }

    float wi_y = (u_out/RES2)*2-1;

    // sample x
    cdf_x += mid*RES2;

    u = fmaxf(samples[0],0.00001);
    left = 0;
    right = RES2-1;
    mid = (left+right)/2;
    mid_v = cdf_x[mid];
    left_v = cdf_x[left];
    right_v = cdf_x[right];

    while (left != mid) { 
         
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_x[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    float pdf_x = cdf_x[mid];
    u_out = float(mid);
    if (mid>0) {
        float p0 = cdf_x[mid-1];
        pdf_x -= p0;
        u_out += fminf((u-p0)/pdf_x,0.9999f);
    } else {
        u_out += fminf(u/pdf_x,0.9999f);
    }
    
    float wi_x = (u_out/RES2)*2-1;

    float v0 = wi_x*wi_x+wi_y*wi_y;
    float v1 = fmaxf(1-v0,0.0f);
    v0 = sqrtf(v0+v1);
    wi_x = wi_x/v0;
    wi_y = wi_y/v0;

    wis[0] = wi_x;
    wis[1] = wi_y;
    pdfs[0] = pdf_x*pdf_y*RES2*RES2/4.0f;
}

__global__ void sample_idx_texture_2d_kernel(
    const int batch_size,
    const int *__restrict__ idx,
    const float *__restrict__ phi,
    const float *__restrict__ cdf_z,
    const float *__restrict__ cdf_y,
    const float *__restrict__ cdf_x,
    const float *__restrict__ samples,
    float *__restrict__ wis,
    float *__restrict__ pdfs
) {
    int batch_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    idx += batch_idx*LOBE;//BXD
    phi += batch_idx*LOBE*2;//BxDx2
    samples += batch_idx*3;//Bx3
    wis += batch_idx*2;//Bx2
    pdfs += batch_idx;//B
    cdf_z += batch_idx*LOBE;//BxD

    // sample lobe
    float u = fmaxf(samples[2],0.00001);
    int left = 0;
    int right = LOBE-1;
    int mid = (left+right)/2;
    float mid_v = cdf_z[mid];
    float left_v = cdf_z[left];
    float right_v = cdf_z[right];

    // if zero lobe, use diffuse sampling
    if (u>right_v) {
        float sin_theta = sqrtf(samples[0]);
        float phi_ = 2*3.141592653589793f*samples[1];

        wis[0] = sin_theta*cosf(phi_);
        wis[1] = sin_theta*sinf(phi_);
        
        // != -1 indicates diffuse sampling
        pdfs[0] = 1.0f/3.141592653589793f;
        return;
    } else {
        // incidates pdf still needs to be sampled
        pdfs[0] = -1;
    }

    while (left != mid) {
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_z[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    int idx_a = idx[mid];
    // AxDxH
    cdf_y += idx_a*LOBE*RES+mid*RES;
    // AxDxHxH
    cdf_x += idx_a*LOBE*RES*RES+mid*RES*RES;
    // BxDx2
    phi += mid*2;


    // sample us
    u = fmaxf(samples[1],0.00001);
    left = 0;
    right = RES-1;
    mid = (left+right)/2;
    mid_v = cdf_y[mid];
    left_v = cdf_y[left];
    right_v = cdf_y[right];
    // find i cdf[i-1] < u <= cdf[i]
    while (left != mid) { 
         
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_y[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    float pdf_y = cdf_y[mid];
    float u_out = float(mid);
    if (mid>0) {
        float p0 = cdf_y[mid-1];
        pdf_y -= p0;
        u_out += fminf((u-p0)/pdf_y,0.9999f);
    } else {
        u_out += fminf(u/pdf_y,0.9999f);
    }

    float wi_y = (u_out/RES)*2-1;

    // sample x
    cdf_x += mid*RES;

    u = fmaxf(samples[0],0.00001);
    left = 0;
    right = RES-1;
    mid = (left+right)/2;
    mid_v = cdf_x[mid];
    left_v = cdf_x[left];
    right_v = cdf_x[right];

    while (left != mid) { 
         
        if (u <= mid_v) {
            right = mid;
            right_v = mid_v;
        } else {
            left = mid;
            left_v = mid_v;
        }
        mid =(left+right)/2;
        mid_v = cdf_x[mid];
    }

    if (u>mid_v) {
        mid += 1;
    }

    pdf_y = cdf_x[mid];
    u_out = float(mid);
    if (mid>0) {
        float p0 = cdf_x[mid-1];
        pdf_y -= p0;
        u_out += fminf((u-p0)/pdf_y,0.9999f);
    } else {
        u_out += fminf(u/pdf_y,0.9999f);
    }
    
    float wi_x = (u_out/RES)*2-1;
    //R.T= [v0,v1]
    //     [-v1,v0]
    float v0 = phi[0];
    float v1 = phi[1];

    // squeeze boundary points to circle
    float x = wi_x*v0 + wi_y*v1;
    float y = wi_y*v0 - wi_x*v1;
    v0 = x*x+y*y;
    v1 = fmaxf(1-v0,0.0f);
    v0 = sqrtf(v0+v1);
    x = x/v0;
    y = y/v0;

    wis[0] = x;
    wis[1] = y;
}

__global__ void fetch_idx_texture_2d_kernel(
    const int batch_size,
    const int *__restrict__ idx,
    const float *__restrict__ phi,
    const float *__restrict__ pdf_z,
    const float *__restrict__ pdf_xy,
    const float *__restrict__ wis,
    float *__restrict__ pdfs
) {
    int batch_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if (batch_idx >= batch_size) {
        return;
    }

    idx += batch_idx*LOBE;//BxD
    phi += batch_idx*LOBE*2;//BxDx2
    pdf_z += batch_idx*LOBE;//BxD
    wis += batch_idx*2;//Bx2
    pdfs += batch_idx;//B

    if (pdfs[0] != -1) {
        // diffuse sampled
        return;
    }
    
    float wi_x = wis[0];
    float wi_y = wis[1];
    float x;
    float y;

    float pdf_out = 0.0f;
    //float weight;
    float v0;
    float v1;
    int idx_a;

    #pragma unroll
    for (int idx_d=0; idx_d < LOBE; idx_d++) {
        //weight = pdf_z[idx_d];
        //if (weight < 1e-3f) {
            // sparsity constrains
        //    continue;
        //}
        idx_a = idx[idx_d];
        v0 = phi[idx_d*2];
        v1 = phi[idx_d*2+1];
        x = v0*wi_x - v1*wi_y;
        y = v1*wi_x + v0*wi_y;

        if (x*x+y*y>1.0f) {
            continue;
        }

        x = (x*0.5f+0.5f)*RES;
        y = (y*0.5f+0.5f)*RES;
        x = fminf(fmaxf(x,0),RES-1);
        y = fminf(fmaxf(y,0),RES-1);
        // D,AxDxHxH
        pdf_out += (pdf_z[idx_d]
                   *pdf_xy[int(x)+int(y)*RES+idx_d*RES*RES+idx_a*RES*RES*LOBE]);
    }
    pdfs[0] = pdf_out*(RES*RES/4.0f);
}

void sample_texture_2d_wrapper(
    const int batch_size,
    const float* cdf_y,
    const float* cdf_x,
    const float* samples,
    float* wis,
    float* pdfs
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sample_texture_2d_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(
        batch_size,
        cdf_y,
        cdf_x,
        samples,
        wis,
        pdfs
    );
    
    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}

void sample_multi_texture_2d_wrapper(
    const int batch_size,
    const float* cdf_z,
    const float* cdf_y,
    const float* cdf_x,
    const float* samples,
    float* wis,
    float* pdfs
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sample_multi_texture_2d_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(
        batch_size,
        cdf_z,
        cdf_y,
        cdf_x,
        samples,
        wis,
        pdfs
    );
    
    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}

void sample_batch_texture_2d_wrapper(
    const int batch_size,
    const float* cdf_y,
    const float* cdf_x,
    const float* samples,
    float* wis,
    float* pdfs
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sample_batch_texture_2d_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(
        batch_size,
        cdf_y,
        cdf_x,
        samples,
        wis,
        pdfs
    );
    
    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}

void sample_idx_texture_2d_wrapper(
    const int batch_size,
    const int* idx,
    const float* phi,
    const float* cdf_z,
    const float* cdf_y,
    const float* cdf_x,
    const float* samples,
    float* wis,
    float* pdfs
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sample_idx_texture_2d_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(
        batch_size,
        idx,
        phi,
        cdf_z,
        cdf_y,
        cdf_x,
        samples,
        wis,
        pdfs
    );
    
    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}

void fetch_idx_texture_2d_wrapper (
    const int batch_size,
    const int* idx,
    const float* phi,
    const float* pdf_z,
    const float* pdf_xy,
    const float* wis,
    float* pdfs
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fetch_idx_texture_2d_kernel<<<NUM_BLOCKS, NUM_THREADS,0,stream>>>(
        batch_size,
        idx,
        phi,
        pdf_z,
        pdf_xy,
        wis,
        pdfs
    );
    
    CUDA_CHECK_ERRORS();
    cudaDeviceSynchronize();
}