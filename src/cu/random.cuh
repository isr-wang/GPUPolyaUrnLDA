#ifndef GPLDA_RANDOM_H
#define GPLDA_RANDOM_H

#include "stdint.h"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

namespace gplda {

__global__ void rng_init(uint32_t seed, uint32_t subsequence, curandStatePhilox4_32_10_t* rng);
__global__ void rng_advance(uint32_t advance, curandStatePhilox4_32_10_t* rng);

}

#endif
