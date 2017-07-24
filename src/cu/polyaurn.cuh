#ifndef GPLDA_POLYAURNSAMPLE_H
#define GPLDA_POLYAURNSAMPLE_H

#include "stdint.h"

namespace gplda {

__global__ void polya_urn_sample(float* Phi, uint32_t* n, float beta, uint32_t V, float** prob, float** alias);
__global__ void polya_urn_transpose(float* Phi);
__global__ void polya_urn_colsums(float* Phi, float* sigma_a, float** prob, uint32_t K);

}

#endif
