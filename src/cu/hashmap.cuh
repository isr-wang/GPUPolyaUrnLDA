#pragma once

#include "types.cuh"
#include "tuning.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#define GPLDA_HASH_EMPTY 0xffffffff00000000

namespace gplda {

template<SynchronizationType sync_type>
struct HashMap {
  u32 size;
  u32 num_elements;
  u64* data;
  u64* stash;
  u64* temp_data;
  u64* temp_stash;
  u32 a[GPLDA_HASH_NUM_FUNCTIONS];
  u32 b[GPLDA_HASH_NUM_FUNCTIONS];
  u32 a_stash;
  u32 b_stash;
  u32 rebuilding;
  curandStatePhilox4_32_10_t* rng;

  __device__ __forceinline__ u32 left_32_bits(u64 x) {
    return (u32) (x >> 32);
  }

  __device__ __forceinline__ u32 right_32_bits(u64 x) {
    return (u32) x;
  }

  __device__ __forceinline__ i32 hash_fn(u64 kv, i32 a, i32 b, i32 slot) {
    i32 key = left_32_bits(kv);
    return ((a * key + b) % 334214459) % slot;
  }

  __device__ __forceinline__ i32 hash_idx(u64 kv, i32 slot, u32* a, u32* b, u32 size) {
    i32 key = left_32_bits(kv);
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
      i32 possible_slot = hash_fn(key, a[i], b[i], size);
      if(possible_slot == slot) {
        return i;
      }
    }
    return 0;
  }


  __device__ inline void init(void* temp, u32 size, curandStatePhilox4_32_10_t* rng) {
    // calculate initialization variables common for all threads
    i32 dim = (sync_type == block) ? blockDim.x : warpSize;
    i32 thread_idx = threadIdx.x % dim;
    u64* data = (u64*) temp;
    u64* stash = data + size*sizeof(u64);

    // set map parameters and calculate random hash functions
    if(thread_idx == 0) {
      this->size = size;
      this->num_elements = 0;
      this->data = data;
      this->stash = stash;
      this->temp_data = this->stash + GPLDA_HASH_STASH_SIZE*sizeof(u64);
      this->temp_stash = this->temp_data + size*sizeof(u64);
      this->rebuilding = false;
      this->rng = rng;
      #pragma unroll
      for(i32 i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
        this->a[i] = __float2uint_rz(size * curand_uniform(this->rng));
        this->b[i] = __float2uint_rz(size * curand_uniform(this->rng));
      }
      this->a_stash = __float2uint_rz(size * curand_uniform(this->rng));
      this->b_stash = __float2uint_rz(size * curand_uniform(this->rng));
    }

    // set map to empty
    for(i32 offset = 0; offset < size / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < size) {
        data[i] = GPLDA_HASH_EMPTY;
      }
    }

    // set stash to empty
    #pragma unroll
    for(i32 offset = 0; offset < GPLDA_HASH_STASH_SIZE / dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < GPLDA_HASH_STASH_SIZE) {
        stash[i] = GPLDA_HASH_EMPTY;
      }
    }

    // synchronize if block map
    if(sync_type == block) {
     __syncthreads();
    }
  }


  __device__ inline void rebuild(u64 kv) {
    // ignore double collisions
    if(this->rebuilding == true) {
      return;
    }

    // synchronize if block map
    if(sync_type == block) {
     __syncthreads();
    }

    // calculate initialization variables common for all threads
    i32 dim = (sync_type == block) ? blockDim.x : warpSize;
    i32 thread_idx = threadIdx.x % dim;

    // first, swap the pointers and generate new hash functions
    if(thread_idx == 0) {
      this->rebuilding = true; // this is thread safe: sync ensures no longer being read
      u64* d = this->data;
      u64* s = this->stash;
      this->data = this->temp_data;
      this->stash = this->temp_stash;
      this->temp_data = d;
      this->temp_stash = s;
      i32 thread_idx = threadIdx.x % this->access_dim;
      #pragma unroll
      for(i32 i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
       this->a[i] = __float2uint_rz(size * curand_uniform(this->rng));
       this->b[i] = __float2uint_rz(size * curand_uniform(this->rng));
      }
      this->a_stash = __float2uint_rz(size * curand_uniform(this->rng));
      this->b_stash = __float2uint_rz(size * curand_uniform(this->rng));
    }

    // synchronize if block map
    if(sync_type == block) {
     __syncthreads();
    }

    // get updated variables
    u32 size = this->size;
    u64* data = this->data;
    u64* stash = this->stash;
    u64* temp_data = this->temp_data;
    u64* temp_stash = this->temp_stash;

    // set map to empty
    for(i32 offset = 0; offset < size / this->access_dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < size) {
        data[i] = GPLDA_HASH_EMPTY;
      }
    }

    // set stash to empty
    #pragma unroll
    for(i32 offset = 0; offset < GPLDA_HASH_STASH_SIZE / this->access_dim + 1; ++offset) {
      i32 i = offset * dim + thread_idx;
      if(i < GPLDA_HASH_STASH_SIZE) {
        stash[i] = GPLDA_HASH_EMPTY;
      }
    }

    // synchronize if block map
    if(sync_type == block) {
      __syncthreads();
    }

    // place keys that collided first
    set(kv);

    // insert elements that were in stash
    #pragma unroll
    for(i32 offset = 0; offset < GPLDA_HASH_STASH_SIZE / this->access_dim + 1; ++offset) {
      i32 i = offset * this->access_dim + thread_idx;
      if(i < GPLDA_HASH_STASH_SIZE) {
        kv = temp_stash[i];
        set(kv);
      }
    }

    // iterate over map and place remaining keys
    for(i32 offset = 0; offset < size / this->access_dim + 1; ++offset) {
      i32 i = offset * this->access_dim + thread_idx;
      if(i < size) {
        kv = temp_data[i];
        set(kv);
      }
    }

    // rebuild is done
    this->rebuilding = false;

    // synchronize if block map
    if(sync_type == block) {
      __syncthreads();
    }
  }


  __device__ __forceinline__ u32 get(u32 key) {
    // check table
    #pragma unroll
    for(i32 i = 0; i < GPLDA_HASH_NUM_FUNCTIONS; ++i) {
      i32 slot = hash_fn(key, this->a[i], this->b[i], this->size);
      u64 kv = this->data[slot];
      if(left_32_bits(kv) == key) {
        return right_32_bits(kv);
      }
    }

    // check stash
    i32 slot = hash_fn(key, this->a_stash, this->b_stash, GPLDA_HASH_STASH_SIZE);
    u64 kv = this->data[slot];
    if(left_32_bits(kv) == key) {
      return right_32_bits(kv);
    }

    // no value: return zero
    return 0;
  }


  __device__ __forceinline__ void set(u64 kv) {
    // get thread-specific variables
    u32* a = this->a;
    u32* b = this->b;
    u64* data = this->data;
    u64* stash = this->stash;
    u32 a_stash = this->a_stash;
    u32 b_stash = this->b_stash;
    u32 size = this->size;

    // repeatedly try to insert values
    u32 current_a = a[0];
    u32 current_b = b[0];
    for(i32 i = 0; i < GPLDA_HASH_MAX_ITERATIONS; ++i) {
      i32 slot = hash_fn(kv, current_a, current_b, size);
      kv = atomicExch(&data[slot],kv);

      // if slot was empty, exit
      if(kv == GPLDA_HASH_EMPTY) {
        break; // don't return: might need to rebuild on that warp
      }

      // determine which hash function was used, try again
      i32 j = hash_idx(kv, slot, a, b, size);
      current_a = a[j+1];
      current_b = b[j+1];
    }

    // if key is still present, try stash
    if(kv != GPLDA_HASH_EMPTY) {
      i32 slot = hash_fn(kv, a_stash, b_stash, GPLDA_HASH_STASH_SIZE);
      kv = atomicExch(&stash[slot], kv);
    }

    // synchronize if block map
    if(sync_type == block) {
      __syncthreads();
    }

    // if key is still present, stash collided, so rebuild table
    if(kv != GPLDA_HASH_EMPTY) {
      rebuild(kv);
    }
  }

  __device__ __forceinline__ void set(u32 key, u32 value) {
    set((u64) key << 32 | value);
  }

  __device__ __forceinline__ void accumulate(u32 key, i32 diff) {
    u32 value = get(key);
  }
};

}
