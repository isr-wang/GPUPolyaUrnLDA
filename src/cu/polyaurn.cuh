#pragma once

#include "types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h> // need to add -lcurand to nvcc flags
#include <cublas_v2.h> // need to add -lcublas to nvcc flags
#include "tuning.cuh"

namespace gplda {

__global__ void polya_urn_init(u32* n, u32* C, f32 beta, u32 V, f32** prob, u32** alias, u32 max_lambda, u32 max_value, curandStatePhilox4_32_10_t* rng);
__global__ void polya_urn_sample(f32* Phi, u32* n, f32 beta, u32 V, f32** prob, u32** alias, u32 max_lambda, u32 max_value, curandStatePhilox4_32_10_t* rng);
void polya_urn_transpose(cudaStream_t* stream, f32* Phi, f32* Phi_temp, u32 K, u32 V, cublasHandle_t* handle, f32* d_zero, f32* d_one);
__global__ void polya_urn_reset(u32* n, u32 V);
__global__ void polya_urn_colsums(f32* Phi, f32* sigma_a, f32 alpha, f32** prob, u32 K);

}
