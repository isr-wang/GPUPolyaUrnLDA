#include "assert.h"
#include "error.cuh"
#include "train.cuh"
#include "spalias.cuh"

namespace gplda {

__host__ __device__ u32 next_pow2(u32 x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return(x);
}

__device__ __forceinline__ u32 warp_lane_offset(u32 lane_bits) {
  return __popc((~(((u32) 4294967295) << (threadIdx.x % 32))) & lane_bits);
}

__device__ __forceinline__ u32 queue_wraparound(u32 idx, u32 queue_size) {
  return idx & (queue_size - 1);
}

__device__ __forceinline__ void warp_queue_pair_push(i32 value, i32 conditional, u32 queue_size,
    i32* q1, u32* q1_read_end, u32* q1_write_end,
    i32* q2, u32* q2_read_end, u32* q2_write_end) {
  // determine which threads write to which queue
  u32 warp_q1_bits = __ballot(conditional);
  u32 warp_q2_bits = __ballot(!conditional); // note: some threads may be inactive
  // determine how many writes are in the warp's view for each queue
  i32 warp_num_q1 = __popc(warp_q1_bits);
  i32 warp_num_q2 = __popc(warp_q2_bits);
  // increment the queue's size, only once per warp, then broadcast to all lanes in the warp
  i32 warp_q1_start;
  i32 warp_q2_start;
  if(threadIdx.x % 32 == 0) {
    warp_q1_start = atomicAdd(q1_write_end, warp_num_q1);
    warp_q2_start = atomicAdd(q2_write_end, warp_num_q2);
  }
  warp_q1_start = __shfl(warp_q1_start, 0);
  warp_q2_start = __shfl(warp_q2_start, 0);
  // if current thread has elements, determine where to write them
  i32* thread_write_queue;
  i32 thread_write_idx;
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

__device__ __forceinline__ i32 warp_queue_pair_pop(/*mut*/ i32* size, u32 queue_size,
    u32* start, u32* end1, u32* end2) {
  i32 read_start;
  i32 read_size;
  // read the queue once per warp
  if(threadIdx.x % warpSize == 0) {
    // first, peek at end1 and end2 to determine how many elements to try to read
    i32 end = min(atomicAdd(end1,0), atomicAdd(end2,0));
    // don't read more than warpSize elements
    read_start = atomicAdd(start,0);
    read_size = min(warpSize, end - read_start);
    // try to read and increment index of elements
    i32 read_success = atomicCAS(start, read_start, read_start + read_size) == read_start;
    read_size = read_success ? read_size : 0;
    // mutate size to indicate how many were actually read
  }
  // broadcast variables to all threads, mutate size to amount read by thread 0
  read_start = __shfl(read_start, 0);
  *size = __shfl(read_size, 0);
  // return start index
  return read_start;
}

__global__ void build_alias(f32** prob, u32** alias, u32 table_size) {
  i32 num_warps = (blockDim.x - 1) / warpSize + 1;
  i32 lane_idx = threadIdx.x % warpSize;
  // determine constants
  f32 cutoff = 1.0f/((f32) table_size);
  // initialize queues
  u32 queue_size = next_pow2(table_size);
  extern __shared__ i32 shared_memory[];
  __shared__ i32 num_active_warps[1];
  __shared__ u32 queue_pair_start[1];
  i32* large = shared_memory;
  __shared__ u32 large_read_end[1];
  __shared__ u32 large_write_end[1];
  i32* small = shared_memory + queue_size;
  __shared__ u32 small_read_end[1];
  __shared__ u32 small_write_end[1];
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
  for(i32 offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    i32 i = threadIdx.x + offset * blockDim.x;
    if(i < table_size) {
      f32 thread_prob = prob[blockIdx.x][i];
      warp_queue_pair_push(i, thread_prob >= cutoff, queue_size, large, large_read_end, large_write_end, small, small_read_end, small_write_end);
    }
  }

  // grab a set of indices from both queues for the warp to work on
  for(i32 warp_num_elements = warpSize; warp_num_elements > 0; /*no increment*/) {
    // try to grab an index, determine how many were grabbed
    warp_num_elements = warpSize; /*may by mutated by warp_queue_pair_pop*/
    i32 warp_queue_idx = warp_queue_pair_pop(&warp_num_elements, queue_size, queue_pair_start, large_read_end, small_read_end);
    // if got an index, fill it
    if(lane_idx < warp_num_elements) {
      i32 thread_large_idx = large[queue_wraparound(warp_queue_idx + lane_idx,queue_size)];
      f32 thread_large_prob = prob[blockIdx.x][thread_large_idx];
      i32 thread_small_idx = small[queue_wraparound(warp_queue_idx + lane_idx,queue_size)];
      f32 thread_small_prob = prob[blockIdx.x][thread_small_idx];
      // determine new, smaller probability and fill the index
      thread_large_prob = (thread_large_prob + thread_small_prob) - cutoff;
      alias[blockIdx.x][thread_small_idx] = thread_large_idx;
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
        for(i32 offset = 0; offset < (large_read_end[0] - queue_pair_start[0]) / warpSize + 1; ++offset) {
          i32 i = queue_pair_start[0] + offset*warpSize + lane_idx;
          if(i < large_read_end[0]) {
            i32 thread_large_idx = large[queue_wraparound(i,queue_size)];
            prob[blockIdx.x][thread_large_idx] = 1.0f;
            alias[blockIdx.x][thread_large_idx] = thread_large_idx;
          }
        }
      } else if(queue_pair_start[0] == large_read_end[0]) {
        // elements still present in small queue
        for(i32 offset = 0; offset < (small_read_end[0] - queue_pair_start[0]) / warpSize + 1; ++offset) {
          i32 i = queue_pair_start[0] + offset*warpSize + lane_idx;
          if(i < small_read_end[0]) {
            i32 thread_small_idx = small[queue_wraparound(i,queue_size)];
            prob[blockIdx.x][thread_small_idx] = 0.0f;
            alias[blockIdx.x][thread_small_idx] = thread_small_idx;
          }
        }
      } else {
        assert(false); // something went wrong, table is incorrect
      }
    }
  }

}

SpAlias::SpAlias(u32 nt, u32 ts) {
  // assign class parameters
  num_tables = nt;
  table_size = ts;
  // allocate array of pointers on host first, so cudaMalloc can populate it
  f32** prob_host = new f32*[num_tables];
  u32** alias_host = new u32*[num_tables];
  // allocate each Alias table
  for(i32 i = 0; i < num_tables; ++i) {
    cudaMalloc(&prob_host[i], table_size * sizeof(f32)) >> GPLDA_CHECK;
    cudaMalloc(&alias_host[i], table_size * sizeof(u32)) >> GPLDA_CHECK;
  }
  // now, allocate array of pointers on device
  cudaMalloc(&prob, num_tables * sizeof(f32*)) >> GPLDA_CHECK;
  cudaMalloc(&alias, num_tables * sizeof(u32*)) >> GPLDA_CHECK;
  // copy array of pointers to device
  cudaMemcpy(prob, prob_host, num_tables * sizeof(f32*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  cudaMemcpy(alias, alias_host, num_tables * sizeof(u32*), cudaMemcpyHostToDevice) >> GPLDA_CHECK;
  // deallocate array of pointers on host
  delete[] prob_host;
  delete[] alias_host;
}

SpAlias::~SpAlias() {
  // allocate array of pointers on host, so we can dereference it
  f32** prob_host = new f32*[num_tables];
  u32** alias_host = new u32*[num_tables];
  // copy array of pointers to host
  cudaMemcpy(prob_host, prob, num_tables * sizeof(f32*), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  cudaMemcpy(alias_host, alias, num_tables * sizeof(u32*), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;
  // free the memory at the arrays being pointed to
  for(i32 i = 0; i < num_tables; ++i) {
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
