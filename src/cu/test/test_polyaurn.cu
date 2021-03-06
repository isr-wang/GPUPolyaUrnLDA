#include "test_polyaurn.cuh"
#include "../poisson.cuh"
#include "../polyaurn.cuh"
#include "../random.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;
using gplda::f32;
using gplda::i32;
using gplda::u32;
using gplda::u64;

namespace gplda_test {

void test_polya_urn_init() {
  constexpr u32 K = 10000;
  constexpr u32 V = 5;
  f32 beta = 0.01;
  u32 n_host[V*K];
  u32 C_host[V] = {1*K, 10*K, 100*K, 1000*K, 10000*K}; // K=5 so E(n_k) = [1, 10, 100, 1000, 10000]
  u32 n_sum[V] = {0,0,0,0,0};
  u32 n_ssq[V] = {0,0,0,0,0};

  u32* n;
  cudaMalloc(&n, K * V * sizeof(u32)) >> GPLDA_CHECK;

  u32* C;
  cudaMalloc(&C, V * sizeof(u32)) >> GPLDA_CHECK;
  cudaMemcpy(C, C_host, V * sizeof(u32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  curandStatePhilox4_32_10_t* Phi_rng;
  cudaMalloc(&Phi_rng, sizeof(curandStatePhilox4_32_10_t)) >> GPLDA_CHECK;
  gplda::rng_init<<<1,1>>>(0,0,Phi_rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  gplda::Poisson* pois = new gplda::Poisson(100, 200, beta);

  gplda::polya_urn_init<<<K,32>>>(n, C, beta, V, pois->pois_alias->prob, pois->pois_alias->alias, pois->max_lambda, pois->max_value, Phi_rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(n_host, n, K * V * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  // check mean by computing sum
  for(i32 j = 0; j < K; ++j) {
    for(i32 i = 0; i < V; ++i) {
      n_sum[i] += n_host[j*V + i];
    }
  }

  // check var by computing sum square
  for(i32 j = 0; j < K; ++j) {
    for(i32 i = 0; i < V; ++i) {
      n_ssq[i] += ((n_host[j*V + i] - (n_sum[i] / K)) * (n_host[j*V + i] - (n_sum[i] / K)));
    }
  }

  assert(n_sum[0] / K <= 2);
  assert(n_sum[1] / K >= 9 && n_sum[1] / K <= 11);
  assert(n_sum[2] / K >= 90 && n_sum[2] / K <= 110);
  assert(n_sum[3] / K >= 900 && n_sum[3] / K <= 1100);
  assert(n_sum[4] / K >= 9000 && n_sum[4] / K <= 11000);

  assert(n_ssq[0] / K <= 2);
  assert(n_ssq[1] / K >= 9 && n_ssq[1] / K <= 11);
  assert(n_ssq[2] / K >= 90 && n_ssq[2] / K <= 110);
  assert(n_ssq[3] / K >= 900 && n_ssq[3] / K <= 1100);
  assert(n_ssq[4] / K >= 9000 && n_ssq[4] / K <= 11000);

  cudaFree(n);
  cudaFree(C);
  cudaFree(Phi_rng);
  delete pois;
}

void test_polya_urn_sample() {
  constexpr u32 V = 3;
  constexpr u32 K = 3;
  f32 tolerance = 0.02f; // large to allow for randomness

  u32 n_host[V*K] = {1,10,100,1,1,1,1000,1000,1000};
  f32 Phi_host[V*K];

  f32* Phi;
  cudaMalloc(&Phi, V*K * sizeof(f32)) >> GPLDA_CHECK;

  u32* n;
  cudaMalloc(&n, V*K * sizeof(u32)) >> GPLDA_CHECK;

  cudaMemcpy(n, n_host, V*K * sizeof(u32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  curandStatePhilox4_32_10_t* Phi_rng;
  cudaMalloc(&Phi_rng, sizeof(curandStatePhilox4_32_10_t)) >> GPLDA_CHECK;
  gplda::rng_init<<<1,1>>>(0,0,Phi_rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  gplda::Poisson* pois = new gplda::Poisson(100, 200, 0.01f);

  gplda::polya_urn_sample<<<K,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(Phi, n, 0.01f, V, pois->pois_alias->prob, pois->pois_alias->alias, pois->max_lambda, pois->max_value, Phi_rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(Phi_host, Phi, V*K * sizeof(f32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(abs(Phi_host[0] - 0.01f) < tolerance);
  assert(abs(Phi_host[1] - 0.09f) < tolerance);
  assert(abs(Phi_host[2] - 0.9f) < tolerance);
  assert(abs(Phi_host[3] - 0.0f) < tolerance || abs(Phi_host[3] - 0.5f) < tolerance || abs(Phi_host[3] - 1.0f) < tolerance);
  assert(abs(Phi_host[4] - 0.0f) < tolerance || abs(Phi_host[4] - 0.5f) < tolerance || abs(Phi_host[4] - 1.0f) < tolerance);
  assert(abs(Phi_host[5] - 0.0f) < tolerance || abs(Phi_host[5] - 0.5f) < tolerance || abs(Phi_host[5] - 1.0f) < tolerance);
  assert(abs(Phi_host[6] - 0.33f) < tolerance);
  assert(abs(Phi_host[7] - 0.33f) < tolerance);
  assert(abs(Phi_host[8] - 0.33f) < tolerance);

  cudaFree(Phi);
  cudaFree(n);
  cudaFree(Phi_rng);
  delete pois;
}

void test_polya_urn_transpose() {
  // 0.3 0.3 0.4
  // 0.2 0.5 0.3
  // 0.1 0.1 0.8
  constexpr u32 V = 3;
  constexpr u32 K = 3;
  f32 Phi_host[V*K] = {0.3f, 0.3f, 0.4f, 0.2f, 0.5f, 0.3f, 0.1f, 0.1f, 0.8f};
  f32* Phi;
  f32* Phi_temp;

  cudaMalloc(&Phi, V*K * sizeof(f32)) >> GPLDA_CHECK;
  cudaMalloc(&Phi_temp, V*K * sizeof(f32)) >> GPLDA_CHECK;

  cudaMemcpy(Phi, Phi_host, V*K * sizeof(f32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  cudaStream_t* stream = new cudaStream_t;
  cudaStreamCreate(stream) >> GPLDA_CHECK;

  cublasHandle_t* cublas_handle = new cublasHandle_t;
  cublasCreate(cublas_handle) >> GPLDA_CHECK;
  cublasSetPointerMode(*cublas_handle, CUBLAS_POINTER_MODE_DEVICE) >> GPLDA_CHECK;

  f32 h_one = 1.0f; // cudaMemset for some reason doesn't work correctly
  f32 h_zero = 0.0f;
  f32* d_zero;
  f32* d_one;
  cudaMalloc(&d_zero, sizeof(f32)) >> GPLDA_CHECK;
  cudaMemcpy(d_zero, &h_zero, sizeof(f32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  cudaMalloc(&d_one, sizeof(f32)) >> GPLDA_CHECK;
  cudaMemcpy(d_one, &h_one, sizeof(f32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  gplda::polya_urn_transpose(stream, Phi, Phi_temp, 3, 3, cublas_handle, d_zero, d_one);
  cudaStreamSynchronize(*stream);

  cudaMemcpy(Phi_host, Phi, V*K * sizeof(f32), cudaMemcpyDeviceToHost);

  assert(Phi_host[0] == 0.3f);
  assert(Phi_host[1] == 0.2f);
  assert(Phi_host[2] == 0.1f);
  assert(Phi_host[3] == 0.3f);
  assert(Phi_host[4] == 0.5f);
  assert(Phi_host[5] == 0.1f);
  assert(Phi_host[6] == 0.4f);
  assert(Phi_host[7] == 0.3f);
  assert(Phi_host[8] == 0.8f);

  cudaStreamDestroy(*stream);
  delete stream;

  cublasDestroy(*cublas_handle);
  delete cublas_handle;

  cudaFree(d_zero);
  cudaFree(d_one);
  cudaFree(Phi);
  cudaFree(Phi_temp);
}

void test_polya_urn_reset() {
  constexpr u32 V = 3;
  constexpr u32 K = 3;
  u32 n_host[V*K] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  u32* n;

  cudaMalloc(&n, V*K * sizeof(u32)) >> GPLDA_CHECK;
  cudaMemcpy(n, n_host, V*K * sizeof(u32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  gplda::polya_urn_reset<<<K, 128>>>(n, V);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(n_host, n, V*K * sizeof(u32), cudaMemcpyDeviceToHost);

  assert(n_host[0] == 0);
  assert(n_host[1] == 0);
  assert(n_host[2] == 0);
  assert(n_host[3] == 0);
  assert(n_host[4] == 0);
  assert(n_host[5] == 0);
  assert(n_host[6] == 0);
  assert(n_host[7] == 0);
  assert(n_host[8] == 0);

  cudaFree(n);
}

void test_polya_urn_colsums() {
  constexpr u32 V = 3;
  constexpr u32 K = 3;
  f32 tolerance = 0.0001f;
  // 0.3 0.3 0.4
  // 0.2 0.5 0.3
  // 0.1 0.1 0.8
  f32 Phi_host[V*K] = {0.3f, 0.2f, 0.1f, 0.3f, 0.5f, 0.1f, 0.4f, 0.3f, 0.8f};
  f32* Phi;

  cudaMalloc(&Phi, V*K * sizeof(f32)) >> GPLDA_CHECK;

  cudaMemcpy(Phi, Phi_host, V*K * sizeof(f32), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  f32* sigma_a;
  cudaMalloc(&sigma_a, V*K * sizeof(f32)) >> GPLDA_CHECK;

  f32** prob;
  cudaMalloc(&prob, V*K * sizeof(f32*)) >> GPLDA_CHECK;

  f32* prob_1;
  f32* prob_2;
  f32* prob_3;
  cudaMalloc(&prob_1, V*K * sizeof(f32)) >> GPLDA_CHECK;
  cudaMalloc(&prob_2, V*K * sizeof(f32)) >> GPLDA_CHECK;
  cudaMalloc(&prob_3, V*K * sizeof(f32)) >> GPLDA_CHECK;

  f32* prob_host[3] = {prob_1, prob_2, prob_3};

  cudaMemcpy(prob, prob_host, V * sizeof(f32*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  gplda::polya_urn_colsums<<<V,GPLDA_POLYA_URN_COLSUMS_BLOCKDIM>>>(Phi, sigma_a, 1.0f, prob, K);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  f32 sigma_a_host[V];

  cudaMemcpy(sigma_a_host, sigma_a, V * sizeof(f32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(sigma_a_host[0] - (0.3f + 0.2f + 0.1f) < tolerance);
  assert(sigma_a_host[1] - (0.3f + 0.5f + 0.1f) < tolerance);
  assert(sigma_a_host[2] - (0.4f + 0.3f + 0.8f) < tolerance);

  f32 prob_host_1[K];
  f32 prob_host_2[K];
  f32 prob_host_3[K];

  cudaMemcpy(prob_host_1, prob_1, K * sizeof(f32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(prob_host_2, prob_2, K * sizeof(f32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(prob_host_3, prob_3, K * sizeof(f32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(abs(prob_host_1[0] - (0.3f / (0.3f + 0.2f + 0.1f))) < tolerance);
  assert(abs(prob_host_1[1] - (0.2f / (0.3f + 0.2f + 0.1f))) < tolerance);
  assert(abs(prob_host_1[2] - (0.1f / (0.3f + 0.2f + 0.1f))) < tolerance);

  assert(abs(prob_host_2[0] - (0.3f / (0.3f + 0.5f + 0.1f))) < tolerance);
  assert(abs(prob_host_2[1] - (0.5f / (0.3f + 0.5f + 0.1f))) < tolerance);
  assert(abs(prob_host_2[2] - (0.1f / (0.3f + 0.5f + 0.1f))) < tolerance);

  assert(abs(prob_host_3[0] - (0.4f / (0.4f + 0.3f + 0.8f))) < tolerance);
  assert(abs(prob_host_3[1] - (0.3f / (0.4f + 0.3f + 0.8f))) < tolerance);
  assert(abs(prob_host_3[2] - (0.8f / (0.4f + 0.3f + 0.8f))) < tolerance);

  cudaFree(Phi) >> GPLDA_CHECK;
  cudaFree(sigma_a) >> GPLDA_CHECK;
  cudaFree(prob) >> GPLDA_CHECK;
  cudaFree(prob_1) >> GPLDA_CHECK;
  cudaFree(prob_2) >> GPLDA_CHECK;
  cudaFree(prob_3) >> GPLDA_CHECK;
}

}
