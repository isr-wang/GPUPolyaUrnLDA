#include "polyaurn.cuh"

namespace gplda {

__global__ void polya_urn_sample(float* Phi, uint32_t* n, uint32_t V) {

}

__global__ void polya_urn_normalize(float* Phi, uint32_t V) {

}

__global__ void polya_urn_colsums(float* Phi, float* sigma_a, uint32_t K) {

}

__global__ void polya_urn_prob(float* Phi, float* sigma_a, uint32_t K, float** prob) {

}

__global__ void reset_sufficient_statistics(uint32_t* n, float* sigma_a, uint32_t V) {

}

}
