#include "test_polyaurn.cuh"
#include "../polyaurn.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;

namespace gplda_test {

void test_polya_urn_init() {

}

void test_polya_urn_sample() {

}

void test_polya_urn_transpose() {
  // 0.3 0.3 0.4
  // 0.2 0.5 0.3
  // 0.1 0.1 0.8
  float Phi_host[9] = {0.3f, 0.3f, 0.4f, 0.2f, 0.5f, 0.3f, 0.1f, 0.1f, 0.8f};
  float* Phi;
  float* Phi_temp;

  cudaMalloc(&Phi, 9 * sizeof(float)) >> GPLDA_CHECK;
  cudaMalloc(&Phi_temp, 9 * sizeof(float)) >> GPLDA_CHECK;

  cudaMemcpy(Phi, Phi_host, 9 * sizeof(float), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  cudaStream_t* stream = new cudaStream_t;
  cudaStreamCreate(stream) >> GPLDA_CHECK;

  cublasHandle_t* cublas_handle = new cublasHandle_t;
  cublasCreate(cublas_handle) >> GPLDA_CHECK;
  cublasSetPointerMode(*cublas_handle, CUBLAS_POINTER_MODE_DEVICE) >> GPLDA_CHECK;

  float* d_zero;
  float* d_one;
  cudaMalloc(&d_zero, sizeof(float)) >> GPLDA_CHECK;
  cudaMemset(d_zero, 0.0f, sizeof(float)) >> GPLDA_CHECK;
  cudaMalloc(&d_one, sizeof(float)) >> GPLDA_CHECK;
  cudaMemset(d_one, 1.0f, sizeof(float)) >> GPLDA_CHECK;

  gplda::polya_urn_transpose(stream, Phi, Phi_temp, 3, 3, cublas_handle, d_zero, d_one);
  cudaStreamSynchronize(*stream);

  cudaMemcpy(Phi_host, Phi, 9 * sizeof(float), cudaMemcpyDeviceToHost);

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

void test_polya_urn_colsums() {
  float tolerance = 0.0001f;
  // 0.3 0.3 0.4
  // 0.2 0.5 0.3
  // 0.1 0.1 0.8
  float Phi_host[9] = {0.3f, 0.2f, 0.1f, 0.3f, 0.5f, 0.1f, 0.4f, 0.3f, 0.8f};
  float* Phi;

  cudaMalloc(&Phi, 9 * sizeof(float)) >> GPLDA_CHECK;

  cudaMemcpy(Phi, Phi_host, 9 * sizeof(float), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  float* sigma_a;
  cudaMalloc(&sigma_a, 3 * sizeof(float)) >> GPLDA_CHECK;

  float** prob;
  cudaMalloc(&prob, 3 * sizeof(float*)) >> GPLDA_CHECK;

  float* prob_1;
  float* prob_2;
  float* prob_3;
  cudaMalloc(&prob_1, 3 * sizeof(float)) >> GPLDA_CHECK;
  cudaMalloc(&prob_2, 3 * sizeof(float)) >> GPLDA_CHECK;
  cudaMalloc(&prob_3, 3 * sizeof(float)) >> GPLDA_CHECK;

  float* prob_host[3] = {prob_1, prob_2, prob_3};

  cudaMemcpy(prob, prob_host, 3 * sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;

  gplda::polya_urn_colsums<<<3,32>>>(Phi, sigma_a, 1.0f, prob, 3);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  float sigma_a_host[3];

  cudaMemcpy(sigma_a_host, sigma_a, 3 * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(sigma_a_host[0] - (0.3f + 0.2f + 0.1f) < tolerance);
  assert(sigma_a_host[1] - (0.3f + 0.5f + 0.1f) < tolerance);
  assert(sigma_a_host[2] - (0.4f + 0.3f + 0.8f) < tolerance);

  float prob_host_1[3];
  float prob_host_2[3];
  float prob_host_3[3];

  cudaMemcpy(prob_host_1, prob_1, 3 * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(prob_host_2, prob_2, 3 * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(prob_host_3, prob_3, 3 * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(prob_host_1[0] - (0.3f / (0.3f + 0.2f + 0.1f)) < tolerance);
  assert(prob_host_1[1] - (0.2f / (0.3f + 0.2f + 0.1f)) < tolerance);
  assert(prob_host_1[2] - (0.1f / (0.3f + 0.2f + 0.1f)) < tolerance);

  assert(prob_host_2[0] - (0.3f / (0.3f + 0.5f + 0.1f)) < tolerance);
  assert(prob_host_2[1] - (0.5f / (0.3f + 0.5f + 0.1f)) < tolerance);
  assert(prob_host_2[2] - (0.1f / (0.3f + 0.5f + 0.1f)) < tolerance);

  assert(prob_host_3[0] - (0.4f / (0.4f + 0.3f + 0.8f)) < tolerance);
  assert(prob_host_3[1] - (0.3f / (0.4f + 0.3f + 0.8f)) < tolerance);
  assert(prob_host_3[2] - (0.8f / (0.4f + 0.3f + 0.8f)) < tolerance);

  cudaFree(Phi) >> GPLDA_CHECK;
  cudaFree(sigma_a) >> GPLDA_CHECK;
  cudaFree(prob) >> GPLDA_CHECK;
  cudaFree(prob_1) >> GPLDA_CHECK;
  cudaFree(prob_2) >> GPLDA_CHECK;
  cudaFree(prob_3) >> GPLDA_CHECK;
}

}
