#pragma once

#include "types.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags
#include "tuning.cuh"

namespace gplda {

__global__ void compute_d_idx(u32* d_len, u32* d_idx, u32 n_docs);

__global__ void warp_sample_topics(u32 size, u32 n_docs,
    u32* z, u32* w, u32* d_len, u32* d_idx, u32* K_d, void* temp,
    u32 K, u32 V, u32 max_K_d,
    f32* Phi_dense,
    f32** prob, u32** alias, curandStatePhilox4_32_10_t* rng);

}
