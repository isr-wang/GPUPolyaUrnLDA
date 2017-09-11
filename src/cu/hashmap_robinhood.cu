//#pragma once

#include "types.cuh"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#include <cstdio>
#include "assert.h"

#define GPLDA_HASH_EMPTY 0xfffff // 20 bits
#define GPLDA_HASH_LINE_SIZE 16
#define GPLDA_HASH_MAX_NUM_LINES 4
#define GPLDA_HASH_NULL_POINTER 0x7f

namespace gplda {

union HashMapEntry {
  #pragma pack(1)
  struct {
    u32 relocate: 1;
    u32 pointer: 7;
    u32 key: 20;
    u64 value: 36;
  };
  u64 int_repr;
};

static_assert(sizeof(HashMapEntry) == sizeof(u64), "#pragma pack(1) failed in HashMapEntry");

struct HashMapEntryRingBuffer {
    u32 start;
    u32 read_end;
    u32 write_end;
    HashMapEntry* buffer;

    __device__ __forceinline__ void push2(HashMapEntry* x, i32 conditional) {

    }

    __device__ __forceinline__ HashMapEntry* pop2(i32 conditional) {
      return 0;
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
  u32 c;
  u32 d;
  u32 needs_rebuild;
  curandStatePhilox4_32_10_t* rng;


  __device__ __forceinline__ i32 hash_slot(u32 key, i32 x, i32 y) {
    return (((x * key + y) % 334214459) % (size / GPLDA_HASH_LINE_SIZE)) * GPLDA_HASH_LINE_SIZE;
  }

  __device__ __forceinline__ i32 key_distance(u32 key, u32 slot) {
    u32 initial_slot = hash_slot(key,a,b);
    u32 stride = hash_slot(key,c,d);
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES - 1; ++i) {
      if((initial_slot + i*stride) % size == slot) {
        return i;
      }
    }
    return GPLDA_HASH_MAX_NUM_LINES;
  }


  __device__ __forceinline__ HashMapEntry entry(u64 ir) {
    HashMapEntry entry;
    entry.int_repr = ir;
    return entry;
  }

  __device__ __forceinline__ HashMapEntry entry(u32 r, u32 p, u32 k, u64 v) {
    HashMapEntry entry;
    entry.relocate = r;
    entry.pointer = p;
    entry.key = k;
    entry.value = v;
    return entry;
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
      float4 r = curand_uniform4(rng);
      a = __float2uint_rz(size * r.w);
      b = __float2uint_rz(size * r.x);
      c = __float2uint_rz(size * r.y);
      d = __float2uint_rz(size * r.z);
    }

    // synchronize to ensure shared memory writes are visible
    sync();

    // set map to empty
    for(i32 offset = 0; offset < size / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < size) {
        data[i] = entry(0,0,GPLDA_HASH_EMPTY,0);
      }
    }

    // set buffer to empty
    for(i32 offset = 0; offset < GPLDA_HASH_LINE_SIZE / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < GPLDA_HASH_LINE_SIZE) {
        buffer[i] = entry(0,0,GPLDA_HASH_EMPTY,0);
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
    i32 initial_slot = hash_slot(key,a,b);
    i32 stride = hash_slot(key,c,d);
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      // compute slot and retrieve entry
      i32 slot = (initial_slot + i * stride) % size;
      HashMapEntry entry = data[slot + half_lane_idx];

      // check if we found the key
      u32 found = __ballot(entry.key == key) & half_lane_mask;
      if(found != 0) {
        return __shfl(entry.value, __ffs(found), warpSize/2);
      }

      // check if there are pointers
      u32 pointer = __ballot(entry.pointer != GPLDA_HASH_NULL_POINTER) & half_lane_mask;
      if(pointer != 0) {
        // TODO: follow pointers
        u32 ptr_found;
        return __shfl(entry.value, __ffs(ptr_found), warpSize/2);
      }

      // check if Robin Hood guarantee indicates no key is present
      u32 no_key = __ballot(entry.key == GPLDA_HASH_EMPTY || key_distance(entry.key, slot) > i) & half_lane_mask;
      if(no_key != 0) {
        return 0;
      }
    }

    // ran out of possible slots: key not present
    return 0;
  }

  __device__ inline void try_accumulate2(u32 key, i32 diff) {
    // determine half warp indices
    i32 lane_idx = threadIdx.x % warpSize;
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / 16) * 4); // 4 if lane >= 16, 0 otherwise

    // build entry to be inserted and shuffle to entire half warp
    HashMapEntry halfwarp_entry = entry(0,0,key,diff);
    halfwarp_entry.int_repr = __shfl(halfwarp_entry.int_repr, 0, warpSize/2);

    // insert key into linked queue
    i32 initial_slot = hash_slot(key,a,b);
    i32 stride = hash_slot(key,c,d);
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      // compute slot
      i32 slot = (initial_slot + i * stride) % size;

      // try to insert, retrying if race condition indicates it is necessary
      i32 retry;
      do {
        // retrieve entry for current half lane, set constants
        HashMapEntry thread_entry = data[slot + half_lane_idx];
        retry = false;

        // determine whether we found the key, an empty slot, or no key is present
        u32 thread_found_key = thread_entry.key == key;
        u32 thread_found_empty = thread_entry.key == GPLDA_HASH_EMPTY;
        u32 thread_no_key = key_distance(thread_entry.key, slot) > i;

        // determine which thread should write
        u32 half_warp_write = __ballot(thread_found_key | thread_found_empty | thread_no_key) & half_lane_mask;
        u32 lane_write_idx = __ffs(half_warp_write) - 1;

        if(lane_idx == lane_write_idx) { // __ffs uses 1-based indexing
          // prepare new entry for table
          HashMapEntry new_entry;

          // determine what kind of new entry we have
          if(thread_found_key == true) {
            // key found: accumulate value
            new_entry = entry(thread_entry.int_repr);
            new_entry.value = new_entry.value + diff;
          } else if(thread_found_empty == true) {
            // empty slot found: insert value
            HashMapEntry new_entry = entry(0,0,key,diff);
          } else if(thread_no_key == true) {
            // TODO: Robin Hood guarantee indicates no key present: insert into eviction queue
            u32 new_pointer;
            // prepare new entry
            new_entry = entry(thread_entry.int_repr);
            new_entry.pointer = new_pointer;
          }

          // swap new and old entry
          u64 old_entry_int_repr = atomicCAS(&thread_entry.int_repr, thread_entry.int_repr, new_entry.int_repr);

          // make sure retrieved entry matches what was expected, so we know that CAS succeeded
          if(old_entry_int_repr != thread_entry.int_repr) {
            // set retry indicator
            retry = true;

            // TODO: clear buffer, if it was requested

          }
        }

        // ensure retry, if necessary, is performed on entire half warp
        retry = __ballot(retry) & half_lane_mask;
      } while(retry != false);
    }

    // resolve queue


  }

  __device__ __forceinline__ void accumulate2(u32 key, i32 diff) {
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
