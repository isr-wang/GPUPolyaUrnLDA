#ifndef GPLDA_POLYAURNSAMPLE_H
#define GPLDA_POLYAURNSAMPLE_H

#include "stdint.h"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

namespace gplda {

__global__ void polya_urn_init(float* Phi, curandStatePhilox4_32_10_t* rng);
__global__ void polya_urn_sample(float* Phi, uint32_t* n, float beta, uint32_t V, float** prob, float** alias, uint32_t max_lambda, uint32_t max_value, curandStatePhilox4_32_10_t* rng);
__global__ void polya_urn_colsums(float* Phi, float* sigma_a, float** prob, uint32_t K);

}

#endif
