#include "assert.h"
#include "error.cuh"
#include "train.cuh"
#include "spalias.cuh"

namespace gplda {

__host__ __device__ uint32_t next_pow2(uint32_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return(x);
}

__device__ __forceinline__ uint32_t warp_lane_offset(uint32_t lane_bits) {
  return __popc((~(((uint32_t) 4294967295) << (threadIdx.x % 32))) & lane_bits);
}

__device__ __forceinline__ uint32_t queue_wraparound(uint32_t idx, uint32_t queue_size) {
  return idx & (queue_size - 1);
}

__device__ __forceinline__ void warp_queue_pair_push(int32_t value, int32_t conditional, uint32_t queue_size,
    int32_t* q1, uint32_t* q1_read_end, uint32_t* q1_write_end,
    int32_t* q2, uint32_t* q2_read_end, uint32_t* q2_write_end) {
  // determine which threads write to which queue
  uint32_t warp_q1_bits = __ballot(conditional);
  uint32_t warp_q2_bits = __ballot(!conditional); // note: some threads may be inactive
  // determine how many writes are in the warp's view for each queue
  int32_t warp_num_q1 = __popc(warp_q1_bits);
  int32_t warp_num_q2 = __popc(warp_q2_bits);
  // increment the queue's size, only once per warp, then broadcast to all lanes in the warp
  int32_t warp_q1_start;
  int32_t warp_q2_start;
  if(threadIdx.x % 32 == 0) {
    warp_q1_start = atomicAdd(q1_write_end, warp_num_q1);
    warp_q2_start = atomicAdd(q2_write_end, warp_num_q2);
  }
  warp_q1_start = __shfl(warp_q1_start, 0);
  warp_q2_start = __shfl(warp_q2_start, 0);
  // if current thread has elements, determine where to write them
  int32_t* thread_write_queue;
  int32_t thread_write_idx;
  if(conditional) {
    thread_write_queue = q1;
    thread_write_idx = warp_q1_start + warp_lane_offset(warp_q1_bits);
  } else {
    thread_write_queue = q2;
    thread_write_idx = warp_q2_start + warp_lane_offset(warp_q2_bits);
  }
  // write elements to both queues
  thread_write_queue[queue_wraparound(thread_write_idx,queue_size)] = value;
  // increment the number of elements that may be read from the queue
  if(threadIdx.x % 32 == 0) {
    // need to do a CAS, otherwise another thread may increment before writing is finished
    do {} while(atomicCAS(q1_read_end, warp_q1_start, warp_q1_start + warp_num_q1) != warp_q1_start);
    do {} while(atomicCAS(q2_read_end, warp_q2_start, warp_q2_start + warp_num_q2) != warp_q2_start);
  }
}

__device__ __forceinline__ int32_t warp_queue_pair_pop(/*mut*/ int32_t* size, uint32_t queue_size,
    uint32_t* start, uint32_t* end1, uint32_t* end2) {
  int32_t read_start;
  int32_t read_size;
  // read the queue once per warp
  if(threadIdx.x % warpSize == 0) {
    // first, peek at end1 and end2 to determine how many elements to try to read
    int32_t end = min(atomicAdd(end1,0), atomicAdd(end2,0));
    // don't read more than warpSize elements
    read_start = atomicAdd(start,0);
    read_size = min(warpSize, end - read_start);
    // try to read and increment index of elements
    int32_t read_success = atomicCAS(start, read_start, read_start + read_size) == read_start;
    read_size = read_success ? read_size : 0;
    // mutate size to indicate how many were actually read
  }
  // broadcast variables to all threads, mutate size to amount read by thread 0
  read_start = __shfl(read_start, 0);
  *size = __shfl(read_size, 0);
  // return start index
  return read_start;
}

__global__ void build_alias(float** prob, float** alias, uint32_t table_size) {
  int32_t num_warps = (blockDim.x - 1) / warpSize + 1;
  int32_t lane_idx = threadIdx.x % warpSize;
  // determine constants
  float cutoff = 1.0f/((float) table_size);
  // initialize queues
  uint32_t queue_size = next_pow2(table_size);
  extern __shared__ int32_t shared_memory[];
  __shared__ int32_t num_active_warps[1];
  __shared__ uint32_t queue_pair_start[1];
  int32_t* large = shared_memory;
  __shared__ uint32_t large_read_end[1];
  __shared__ uint32_t large_write_end[1];
  int32_t* small = shared_memory + queue_size;
  __shared__ uint32_t small_read_end[1];
  __shared__ uint32_t small_write_end[1];
  if(threadIdx.x == 0) {
    num_active_warps[0] = num_warps;
    queue_pair_start[0] = 0;
    large_read_end[0] = 0;
    small_read_end[0] = 0;
    large_write_end[0] = 0;
    small_write_end[0] = 0;
  }
  __syncthreads();

  // loop over PMF, build large queue
  for(int32_t offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    int32_t i = threadIdx.x + offset * blockDim.x;
    if(i < table_size) {
      float thread_prob = prob[blockIdx.x][i];
      warp_queue_pair_push(i, thread_prob >= cutoff, queue_size, large, large_read_end, large_write_end, small, small_read_end, small_write_end);
    }
  }

  // grab a set of indices from both queues for the warp to work on
  for(int32_t warp_num_elements = warpSize; warp_num_elements > 0; /*no increment*/) {
    // try to grab an index, determine how many were grabbed
    warp_num_elements = warpSize; /*may by mutated by warp_queue_pair_pop*/
    int32_t warp_queue_idx = warp_queue_pair_pop(&warp_num_elements, queue_size, queue_pair_start, large_read_end, small_read_end);
    // if got an index, fill it
    if(lane_idx < warp_num_elements) {
      int32_t thread_large_idx = large[queue_wraparound(warp_queue_idx + lane_idx,queue_size)];
      float thread_large_prob = prob[blockIdx.x][thread_large_idx];
      int32_t thread_small_idx = small[queue_wraparound(warp_queue_idx + lane_idx,queue_size)];
      float thread_small_prob = prob[blockIdx.x][thread_small_idx];
      // determine new, smaller probability and fill the index
      thread_large_prob = (thread_large_prob + thread_small_prob) - cutoff;
      alias[blockIdx.x][thread_small_idx] = (float) thread_large_idx;
      prob[blockIdx.x][thread_large_idx] = thread_large_prob;
      // finally, push remaining values back onto queues
      warp_queue_pair_push(thread_large_idx, thread_large_prob >= cutoff, queue_size, large, large_read_end, large_write_end, small, small_read_end, small_write_end);
    }
  }

  // at this point, one of the queues must be empty, so finish remaining values using slowest warp
  if(lane_idx == 0) {
    // determine if last exiting warp
    if(atomicSub(num_active_warps, 1) == 1) {
      // final warp: set remaining probabilities to 1
      if(queue_pair_start[0] == small_read_end[0]) {
        // elements still present in large queue
        for(int32_t offset = 0; offset < (large_read_end[0] - queue_pair_start[0]) / warpSize + 1; ++offset) {
          int32_t i = queue_pair_start[0] + offset*warpSize + lane_idx;
          if(i < large_read_end[0]) {
            int32_t thread_large_idx = large[queue_wraparound(i,queue_size)];
            prob[blockIdx.x][thread_large_idx] = 1.0f;
            alias[blockIdx.x][thread_large_idx] = (float) thread_large_idx;
          }
        }
      } else if(queue_pair_start[0] == large_read_end[0]) {
        // elements still present in small queue
        for(int32_t offset = 0; offset < (small_read_end[0] - queue_pair_start[0]) / warpSize + 1; ++offset) {
          int32_t i = queue_pair_start[0] + offset*warpSize + lane_idx;
          if(i < small_read_end[0]) {
            int32_t thread_small_idx = small[queue_wraparound(i,queue_size)];
            prob[blockIdx.x][thread_small_idx] = 0.0f;
            alias[blockIdx.x][thread_small_idx] = (float) thread_small_idx;
          }
        }
      } else {
        assert(false); // something went wrong, table is incorrect
      }
    }
  }
}

SpAlias::SpAlias(uint32_t nt, uint32_t ts) {
  // assign class parameters
  num_tables = nt;
  table_size = ts;
  // allocate array of pointers on host first, so cudaMalloc can populate it
  float** prob_host = new float*[num_tables];
  float** alias_host = new float*[num_tables];
  // allocate each Alias table
  for(int32_t i = 0; i < num_tables; ++i) {
    cudaMalloc(&prob_host[i], table_size * sizeof(float)) >> GPLDA_CHECK;
    cudaMalloc(&alias_host[i], table_size * sizeof(float)) >> GPLDA_CHECK;
  }
  // now, allocate array of pointers on device
  cudaMalloc(&prob, num_tables * sizeof(float*)) >> GPLDA_CHECK;
  cudaMalloc(&alias, num_tables * sizeof(float*)) >> GPLDA_CHECK;
  // copy array of pointers to device
  cudaMemcpy(prob, prob_host, num_tables * sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  cudaMemcpy(alias, alias_host, num_tables * sizeof(float*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  // deallocate array of pointers on host
  delete[] prob_host;
  delete[] alias_host;
}

SpAlias::~SpAlias() {
  // allocate array of pointers on host, so we can dereference it
  float** prob_host = new float*[num_tables];
  float** alias_host = new float*[num_tables];
  // copy array of pointers to host
  cudaMemcpy(prob_host, prob, num_tables * sizeof(float*), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(alias_host, alias, num_tables * sizeof(float*), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  // free the memory at the arrays being pointed to
  for(int32_t i = 0; i < num_tables; ++i) {
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
