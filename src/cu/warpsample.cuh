#pragma once

#include "stdint.h"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags
#include "tuning.cuh"

namespace gplda {

__global__ void compute_d_idx(uint32_t* d_len, uint32_t* d_idx, uint32_t n_docs);

__global__ void warp_sample_topics(uint32_t size, uint32_t n_docs, uint32_t *z, uint32_t *w, uint32_t *d_len, uint32_t *d_idx, float** prob, uint32_t** alias, curandStatePhilox4_32_10_t* rng);

}
