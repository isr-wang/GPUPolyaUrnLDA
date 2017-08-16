#pragma once

#include "types.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

namespace gplda {

__global__ void rng_init(u32 seed, u32 subsequence, curandStatePhilox4_32_10_t* rng);
__global__ void rng_advance(u32 advance, curandStatePhilox4_32_10_t* rng);

}
