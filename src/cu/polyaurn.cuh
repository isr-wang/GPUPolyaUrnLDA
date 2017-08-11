#pragma once

#include "stdint.h"
#include <cuda_runtime.h>
#include <curand_kernel.h> // need to add -lcurand to nvcc flags
#include <cublas_v2.h> // need to add -lcublas to nvcc flags
#include "tuning.cuh"

namespace gplda {

__global__ void polya_urn_init(uint32_t* n, uint32_t* C, float beta, uint32_t V, float** prob, uint32_t** alias, uint32_t max_lambda, uint32_t max_value, curandStatePhilox4_32_10_t* rng);
__global__ void polya_urn_sample(float* Phi, uint32_t* n, float beta, uint32_t V, float** prob, uint32_t** alias, uint32_t max_lambda, uint32_t max_value, curandStatePhilox4_32_10_t* rng);
void polya_urn_transpose(cudaStream_t* stream, float* Phi, float* Phi_temp, uint32_t K, uint32_t V, cublasHandle_t* handle, float* d_zero, float* d_one);
__global__ void polya_urn_reset(uint32_t* n, uint32_t V);
__global__ void polya_urn_colsums(float* Phi, float* sigma_a, float alpha, float** prob, uint32_t K);

}
