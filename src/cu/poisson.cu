#include "assert.h"
#include "error.cuh"
#include "poisson.cuh"
#include "train.cuh"

namespace gplda {


__device__ __forceinline__ unsigned int warp_lane_id_bits(int thread_idx) {
  return ((unsigned int) 1) << (thread_idx % warpSize);
}

__device__ __forceinline__ unsigned int warp_lane_offset(unsigned int lane_bits, int thread_idx) {
  return __popc((~(((unsigned int) 4294967295) << (thread_idx % warpSize))) & lane_bits);
}

__device__ __forceinline__ int queue_pop(int* queue, int* start, int* end) {
  return 1;
}

__device__ __forceinline__ void queue_push(int value, int* queue, int* start, int* end) {
  return;
}

__global__ void build_poisson(float** prob, float** alias, float beta, int table_size) {
  assert(blockDim.x % 32 == 0); // kernel will fail if warpSize != 32
  int warp_idx = threadIdx.x / warpSize;
  int num_warps = blockDim.x / warpSize + 1;
  int lane_idx = threadIdx.x % warpSize;
  // determine constants
  int lambda = blockIdx.x; // each block builds one table
  float L = lambda + beta;
  float cutoff = 1.0/((float) table_size);
  // populate PMF
  for(int offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    int i = threadIdx.x + offset * blockDim.x;
    float x = i;
    if(i < table_size) {
      prob[lambda][i] = expf(x*logf(L) - L - lgammaf(x + 1));
    }
  }
  __syncthreads();
  // initialize queues
  __shared__ int num_active_warps[1];
  /*extern*/ __shared__ int large[200];
  __shared__ int large_start[1];
  __shared__ int large_end[1];
  /*extern*/ __shared__ int small[200];
  __shared__ int small_start[1];
  __shared__ int small_end[1];
  if(threadIdx.x == 0) {
    num_active_warps[0] = num_warps;
    large_start[0] = 0;
    large_end[0] = 0;
    small_start[0] = 0;
    small_end[0] = 0;
  }
  __syncthreads();
  // loop over PMF, build large queue
  for(int offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    int i = threadIdx.x + offset * blockDim.x;
    if(i < table_size) {
      float thread_prob = prob[lambda][i];
      // determine which threads have large probabilities
      unsigned int warp_large_bits = __ballot(thread_prob >= cutoff);
      unsigned int warp_small_bits = __ballot(thread_prob < cutoff); // don't negate: some threads may be inactive
      // determine how many large probabilities are in the warp's view
      int warp_num_large = __popc(warp_large_bits);
      int warp_num_small = __popc(warp_small_bits);
      // increment the array's size, only once per warp, then broadcast to all lanes in the warp
      int warp_large_start;
      int warp_small_start;
      if(lane_idx == 0) {
        warp_large_start = atomicAdd(large_end, warp_num_large);
        warp_small_start = atomicAdd(small_end, warp_num_small);
      }
      warp_large_start = __shfl(warp_large_start, 0);
      warp_small_start = __shfl(warp_small_start, 0);
      // if current warp has elements, add elements to the array
      if(thread_prob >= cutoff) {
        large[warp_large_start + warp_lane_offset(warp_large_bits, threadIdx.x)] = i;
      } else {
        small[warp_small_start + warp_lane_offset(warp_small_bits, threadIdx.x)] = i;
      }
    }
  }
  __syncthreads();
  // grab a set of indices from large queue for the thread to work on
  int thread_large_idx = -1; /* flag value: -1 means empty */
  float thread_large_prob;
  while(true) {
    // if needed, grab indices from large queue for the warp to work on
    if(thread_large_idx < 0) {
      // try to grab an index
      thread_large_idx = queue_pop(large, large_start, large_end);
      // if got an index, get its value
      if(thread_large_idx >= 0) {
        thread_large_prob = prob[lambda][thread_large_idx];
      }
    }
    // if holding a large index, try to fill a small index
    if(thread_large_idx >= 0) {
      // try to grab an index from small queue
      int thread_small_idx = queue_pop(small, small_start, small_end);
      // if got an index, fill it
      if(thread_small_idx >= 0) {
        float thread_small_prob = prob[lambda][thread_small_idx];
        thread_large_prob = (thread_large_prob + thread_small_prob) - 1.0;
        alias[lambda][thread_small_idx] = thread_large_idx;
        // check if large probability became small
        if(thread_large_prob < cutoff) {
          // make prob small
          prob[lambda][thread_large_idx] = thread_large_prob;
          // add to small queue
          queue_push(thread_large_idx, small, small_start, small_end);
          thread_large_idx = -1;
        }
      } else {
        // if small queue is empty, push value back onto large queue, and exit
        queue_push(thread_large_idx, large, large_start, large_end);
        thread_large_idx = -1;
        break;
      }
    } else {
      // large queue is empty, exit
      break;
    }
  }
  // at this point, both queues should now be near empty, so finish them using one warp
  if(warp_idx != 0) {
    atomicSub(num_active_warps, 1);
    return;
  } else {
    do {/*wait*/} while(num_active_warps[0] > 1);
    printf("Finishing remaining values");
  }
}


__global__ void draw_poisson(float** prob, float** alias, int* lambda, int n) {
}

Poisson::Poisson(int ml, int mv) {
  // assign class parameters
  max_lambda = ml;
  max_value = mv;
  // allocate array of pointers on host first, so cudaMalloc can populate it
  float** prob_host = new float*[max_lambda];
  float** alias_host = new float*[max_lambda];
  // allocate each Alias table
  for(size_t i = 0; i < max_lambda; ++i) {
    cudaMalloc(&prob_host[i], max_value * sizeof(float)) >> GPLDA_CHECK;
    cudaMalloc(&alias_host[i], max_value * sizeof(float)) >> GPLDA_CHECK;
  }
  // now, allocate array of pointers on device
  cudaMalloc(&prob, max_lambda * sizeof(float*)) >> GPLDA_CHECK;
  cudaMalloc(&alias, max_lambda * sizeof(float*)) >> GPLDA_CHECK;
  // copy array of pointers to device
  cudaMemcpy(prob, prob_host, max_lambda * sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  cudaMemcpy(alias, alias_host, max_lambda * sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  // deallocate array of pointers on host
  delete[] prob_host;
  delete[] alias_host;
  // launch kernel to build the alias tables
  build_poisson<<</*max_lambda*/1,96/*32*/,max_value*sizeof(int)>>>(prob, alias, ARGS->beta, max_value);
  cudaDeviceSynchronize();
}

Poisson::~Poisson() {
  // allocate array of pointers on host, so we can dereference it
  float** prob_host = new float*[max_lambda];
  float** alias_host = new float*[max_lambda];
  // copy array of pointers to host
  cudaMemcpy(prob_host, prob, max_lambda * sizeof(float*), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(alias_host, alias, max_lambda * sizeof(float*), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  // free the memory at the arrays being pointed to
  for(size_t i = 0; i < max_lambda; ++i) {
    cudaFree(prob_host[i]) >> GPLDA_CHECK;
    cudaFree(alias_host[i]) >> GPLDA_CHECK;
  }
  // free the memory of the pointer array on device
  cudaFree(prob) >> GPLDA_CHECK;
  cudaFree(alias) >> GPLDA_CHECK;
  // deallocate array of pointers on host
  delete[] prob_host;
  delete[] alias_host;
}

}
