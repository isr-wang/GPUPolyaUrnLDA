//#pragma once

#include "types.cuh"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#include <cstdio>
#include "assert.h"

#define GPLDA_HASH_EMPTY 0xfffff // 20 bits
#define GPLDA_HASH_LINE_SIZE 16
#define GPLDA_HASH_MAX_NUM_LINES 6

namespace gplda {

union HashMapEntry {
  struct {
    u32 relocate: 1;
    u32 backpointer_hash: 3;
    u32 backpointer_slot: 4;
    u32 key: 20;
    u64 value: 36;
  };
  u64 int_repr;
  HashMapEntry(u32 r, u32 bh, u32 bs, u32 k, u64 v) {
    this->relocate = r;
    this->backpointer_hash = bh;
    this->backpointer_slot = bs;
    this->key = k;
    this->value = v;
  }
};

template<SynchronizationType sync_type>
struct HashMap {
  u32 size;
  u32 max_size;
  HashMapEntry* data;
  HashMapEntry* temp_data;
  HashMapEntry* buffer;
  u32 a;
  u32 b;
  u32 c[GPLDA_HASH_MAX_NUM_LINES - 1];
  u32 needs_rebuild;
  curandStatePhilox4_32_10_t* rng;

  __device__ __forceinline__ u32 left_32_bits(u64 x) {
    return (u32) (x >> 32);
  }

  __device__ __forceinline__ u32 right_32_bits(u64 x) {
    return (u32) x;
  }

  __device__ __forceinline__ i32 hash_fn(u32 key) {
    return (a * key + b) % 334214459;
  }

  __device__ __forceinline__ i32 hash_slot(u32 key) {
    return (hash_fn(key) % (size / GPLDA_HASH_LINE_SIZE)) * GPLDA_HASH_LINE_SIZE;
  }

  __device__ __forceinline__ i32 rev_hash_fn(u32 key, i32 i) {
    return i == 0 ? hash_fn(key) : key ^ c[(((c[0] * key + c[1]) % 334214459) + i - 1) % (GPLDA_HASH_MAX_NUM_LINES - 1)];
  }

  __device__ __forceinline__ i32 rev_hash_fn_idx(u32 key, u32 slot) {
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      if(rev_hash_fn(key, i) == slot) {
        return i;
      }
    }
  }



  __device__ __forceinline__ void sync() {
    if(sync_type == block) {
      __syncthreads();
    }
  }

  __device__ inline void provide_buffer(u64* in_buffer) {
    if(threadIdx.x == 0) {
      buffer = (HashMapEntry*) in_buffer;
    }
    sync();
  }




  __device__ inline void init(void* in_data, u32 in_size, u32 in_max_size, curandStatePhilox4_32_10_t* in_rng) {
    // calculate initialization variables common for all threads
    i32 dim = (sync_type == block) ? blockDim.x : warpSize;
    i32 thread_idx = threadIdx.x % dim;

    // set map parameters and calculate random hash functions
    if(thread_idx == 0) {
      // round down to ensure cache alignment
      max_size = (in_max_size / GPLDA_HASH_LINE_SIZE) * GPLDA_HASH_LINE_SIZE;
      size = min((in_size / GPLDA_HASH_LINE_SIZE + 1) * GPLDA_HASH_LINE_SIZE, in_max_size);

      // perform pointer arithmetic
      data = (HashMapEntry*) in_data;
      temp_data = data + max_size; // no sizeof for typed pointer arithmetic
      buffer = temp_data + max_size; // no sizeof for typed pointer arithmetic

      needs_rebuild = 0;
      rng = in_rng; // make sure this->rng is set before use
      a = __float2uint_rz(size * curand_uniform(rng));
      b = __float2uint_rz(size * curand_uniform(rng));
      #pragma unroll
      for(i32 i = 1; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
        c[i-1] = __float2uint_rz(size * curand_uniform(rng));
      }
    }

    // synchronize to ensure shared memory writes are visible
    sync();

    // set map to empty
    for(i32 offset = 0; offset < size / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < size) {
        data[i] = HashMapEntry(0,0,0,GPLDA_HASH_EMPTY,0);
      }
    }

    // set buffer to empty
    for(i32 offset = 0; offset < GPLDA_HASH_LINE_SIZE / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < GPLDA_HASH_LINE_SIZE) {
        buffer[i] = HashMapEntry(0,0,0,GPLDA_HASH_EMPTY,0);
      }
    }

    // synchronize to ensure initialization is complete
    sync();
  }





  __device__ inline void rebuild() {

  }





  __device__ inline u32 get2(u32 key) {
    // shuffle key to entire half-warp
    key = __shfl(key, 0, warpSize/2);
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / 16) * 4); // 4 if lane >= 16, 0 otherwise

    // check table
    i32 initial_slot = hash_slot(key);
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      // compute slot
      i32 slot = rev_hash_fn(initial_slot, i);

      HashMapEntry entry = data[slot + half_lane_idx];

      // check if we found the key
      u32 found = __ballot(entry.key == key) & half_lane_mask;
      if(found != 0) {
        return __shfl(entry.value, __ffs(found), warpSize/2);
      }

      // check if Robin Hood guarantee indicates no key is present
      u32 no_key = __ballot(entry.key == GPLDA_HASH_EMPTY || rev_hash_fn_idx(entry.key, slot) > i) & half_lane_mask;
      if(no_key != 0) {
        return 0;
      }
    }

    // ran out of possible slots: key not present
    return 0;
  }

  __device__ inline void try_accumulate2(u32 key, u32 diff) {
    // shuffle key and diff to entire half warp
    key = __shfl(key, 0, warpSize/2);
    diff = __shfl(diff, 0, warpSize/2);
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    i32 half_warp_idx = threadIdx.x / (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / 16) * 4); // 4 if lane >= 16, 0 otherwise
    i32 ring_buffer_start = 0;
    i32 backpointer_hash;
    i32 backpointer_slot;

    // insert key into buffer
    if(half_lane_idx == 0) {
      backpointer_hash = 1; // buffer
      backpointer_slot = (ring_buffer_start + half_warp_idx) % GPLDA_HASH_MAX_NUM_LINES;
      buffer[backpointer_slot] = HashMapEntry(0,0,0,key,diff);
    }

    // forward pass to find empty value, accumulate key if present

    // backward pass to insert value

  }

  __device__ __forceinline__ void accumulate2(u32 key, u32 diff) {
    // try to accumulate
    try_accumulate2(key, diff);

    // rebuild if too large
    sync();
    if(needs_rebuild == 1) {
      rebuild();
    }
  }
};

}
