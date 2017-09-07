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
  #pragma pack(1)
  struct {
    u32 relocate: 1;
    u32 backpointer_hash: 3;
    u32 backpointer_idx: 4;
    u32 key: 20;
    u64 value: 36;
  };
  u64 int_repr;
  HashMapEntry(u64 ir) {
    this->int_repr = ir;
  }
  HashMapEntry(u32 r, u32 bh, u32 bs, u32 k, u64 v) {
    this->relocate = r;
    this->backpointer_hash = bh;
    this->backpointer_idx = bs;
    this->key = k;
    this->value = v;
  }
};

static_assert(sizeof(HashMapEntry) == sizeof(u64), "#pragma pack(1) failed in HashMapEntry");

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

  __device__ __forceinline__ i32 hash_fn(u32 key) {
    return (a * key + b) % 334214459;
  }

  __device__ __forceinline__ i32 hash_slot(u32 key) {
    return (hash_fn(key) % (size / GPLDA_HASH_LINE_SIZE)) * GPLDA_HASH_LINE_SIZE;
  }

  __device__ __forceinline__ i32 cipher_hash_fn(u32 key) {
    return (c[0] * key + c[1]) % 334214459;
  }

  __device__ __forceinline__ i32 cipher_fn(i32 slot, i32 fn_idx) {
    return fn_idx == 0 ? slot : slot ^ cipher_hash_fn(fn_idx - 1);
  }

  __device__ __forceinline__ i32 cipher_fn_idx(i32 key, u32 slot) {
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_MAX_NUM_LINES; ++i) {
      if(cipher_fn(i, key) == slot) {
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
      // compute slot and retrieve entry
      i32 slot = cipher_fn(initial_slot, i);
      HashMapEntry entry = data[slot + half_lane_idx];

      // check if we found the key
      u32 found = __ballot(entry.key == key) & half_lane_mask;
      if(found != 0) {
        return __shfl(entry.value, __ffs(found), warpSize/2);
      }

      // check if Robin Hood guarantee indicates no key is present
      u32 no_key = __ballot(entry.key == GPLDA_HASH_EMPTY || cipher_fn_idx(entry.key, slot) > i) & half_lane_mask;
      if(no_key != 0) {
        return 0;
      }
    }

    // ran out of possible slots: key not present
    return 0;
  }

  __device__ inline void try_accumulate2(u32 key, i32 diff) {
    // determine half warp indices
    i32 half_lane_idx = threadIdx.x % (warpSize / 2);
    i32 half_warp_idx = threadIdx.x / (warpSize / 2);
    u32 half_lane_mask = 0x0000ffff << (((threadIdx.x % warpSize) / 16) * 4); // 4 if lane >= 16, 0 otherwise

    // acquire ring buffer location
    i32 ring_buffer_start = 0;

    // build entry to be inserted and shuffle to entire half warp
    HashMapEntry halfwarp_entry = HashMapEntry(0,1,0,key,diff);
    halfwarp_entry.int_repr = __shfl(halfwarp_entry.int_repr, 0, warpSize/2);

    // insert key into buffer
    if(half_lane_idx == 0) {
      i32 buffer_idx = (ring_buffer_start + half_warp_idx) % GPLDA_HASH_LINE_SIZE;
      buffer[buffer_idx] = halfwarp_entry;
      halfwarp_entry.backpointer_hash = 1; // buffer
      halfwarp_entry.backpointer_idx = buffer_idx;
    }

    // forward pass: find empty value, accumulate key if present
    i32 initial_slot = hash_slot(key);
    i32 slot_idx = 0;
    i32 done = false;
    for(i32 i = 0; i < 7 * (32 - __clz(size)); ++i) { // fast log base 2
      // compute slot and retrieve entry
      i32 slot = cipher_fn(initial_slot, slot_idx);
      HashMapEntry thread_entry = data[slot + half_lane_idx];

      // assuming Robin Hood guarantees have not kicked in yet, check if we found the key
      if(halfwarp_entry.backpointer_hash == 1) {
        if(thread_entry.key == key) {
          // key found: set relocate intention, accumulate, clear buffer, and exit if successful
          buffer[halfwarp_entry.backpointer_idx].relocate = true;
          HashMapEntry replacement = thread_entry;
          replacement.value += diff;
          // perform CAS, retrying if necessary
          while(true) {
            HashMapEntry old = HashMapEntry(atomicCAS(&data[slot + half_lane_idx].int_repr, thread_entry.int_repr, replacement.int_repr));
            if(old.int_repr == thread_entry.int_repr) {
              // update was successful: clear buffer
              buffer[halfwarp_entry.backpointer_idx] = HashMapEntry(0,0,0,GPLDA_HASH_EMPTY,0);
              done = true;
              break;
            } else if(old.key != thread_entry.key) {
              // key and value changed: remove relocate intention
              buffer[halfwarp_entry.backpointer_idx].relocate = false;
              break;
            }
            // else value changed but key didn't: try another CAS
          }
        }
        if((__ballot(done == true) & half_lane_mask) != 0) {
          return;
        }
      }

      // key is not present: see if we can take some other key's slot
      i32 thread_entry_initial_slot;
      if(thread_entry.backpointer_hash == 0) {
        // no backpointer: reverse current entry's hash function
        thread_entry_initial_slot = cipher_fn_idx(thread_entry.key, slot);
      } else {
        // follow backpointer, then reverse the hash function
        HashMapEntry thread_backpointer_entry = thread_entry.backpointer_hash == 1 ?
            buffer[thread_entry.backpointer_idx] : // backpointer points to buffer
            data[cipher_fn(slot, thread_entry.backpointer_hash) + thread_entry.backpointer_idx]; // backpointer points to table
        thread_entry_initial_slot = cipher_fn_idx(thread_backpointer_entry.key, slot);
      }

      // increment slot index
      slot_idx++;
      if(slot_idx >= GPLDA_HASH_MAX_NUM_LINES) {
        // no available slot in 6 cache lines: resize table
      }
    }

    // backward pass to insert value
    while(true) {
      break;
    }


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
