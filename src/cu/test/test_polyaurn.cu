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

void test_polya_urn_colsums() {
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

  assert(sigma_a_host[0] - (0.3 + 0.2 + 0.1) < 0.0001);
  assert(sigma_a_host[1] - (0.3 + 0.5 + 0.1) < 0.0001);
  assert(sigma_a_host[2] - (0.4 + 0.3 + 0.8) < 0.0001);

  float prob_host_1[3];
  float prob_host_2[3];
  float prob_host_3[3];

  cudaMemcpy(prob_host_1, prob_1, 3 * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(prob_host_2, prob_2, 3 * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(prob_host_3, prob_3, 3 * sizeof(float), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  assert(prob_host_1[0] - (0.3 / (0.3 + 0.2 + 0.1)) < 0.0001);
  assert(prob_host_1[1] - (0.2 / (0.3 + 0.2 + 0.1)) < 0.0001);
  assert(prob_host_1[2] - (0.1 / (0.3 + 0.2 + 0.1)) < 0.0001);

  assert(prob_host_2[0] - (0.3 / (0.3 + 0.5 + 0.1)) < 0.0001);
  assert(prob_host_2[1] - (0.5 / (0.3 + 0.5 + 0.1)) < 0.0001);
  assert(prob_host_2[2] - (0.1 / (0.3 + 0.5 + 0.1)) < 0.0001);

  assert(prob_host_3[0] - (0.4 / (0.4 + 0.3 + 0.8)) < 0.0001);
  assert(prob_host_3[1] - (0.3 / (0.4 + 0.3 + 0.8)) < 0.0001);
  assert(prob_host_3[2] - (0.8 / (0.4 + 0.3 + 0.8)) < 0.0001);

  cudaFree(Phi) >> GPLDA_CHECK;
  cudaFree(sigma_a) >> GPLDA_CHECK;
  cudaFree(prob) >> GPLDA_CHECK;
  cudaFree(prob_1) >> GPLDA_CHECK;
  cudaFree(prob_2) >> GPLDA_CHECK;
  cudaFree(prob_3) >> GPLDA_CHECK;
}

}
