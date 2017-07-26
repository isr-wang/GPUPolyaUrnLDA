#ifndef GPLDA_WARPSAMPLER_H
#define GPLDA_WARPSAMPLER_H

#include "stdint.h"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

namespace gplda {

void compute_d_idx(cudaStream_t& stream, uint32_t* d_len, uint32_t* d_idx, uint32_t n_docs);

__global__ void warp_sample_topics(uint32_t size, uint32_t n_docs, uint32_t *z, uint32_t *w, uint32_t *d_len, uint32_t *d_idx, curandStatePhilox4_32_10_t* rng);

}

#endif
