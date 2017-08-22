#include "test_hashmap.cuh"
#include "../hashmap.cuh"
#include "../random.cuh"
#include "../error.cuh"
#include "assert.h"

using gplda::FileLine;
using gplda::f32;
using gplda::i32;
using gplda::u32;
using gplda::u64;

namespace gplda_test {

template<gplda::SynchronizationType sync_type>
__global__ void test_hash_map_init(void* map_storage, u32 size, curandStatePhilox4_32_10_t* rng) {
  __shared__ gplda::HashMap<sync_type> m[1];
  m->init(map_storage, size, size, rng);
}



template<gplda::SynchronizationType sync_type, i32 rebuild>
__global__ void test_hash_map_insert(void* map_storage, u32 size, u32 num_elements, u32* out, curandStatePhilox4_32_10_t* rng) {
  __shared__ gplda::HashMap<sync_type> m[1];
  m->init(map_storage, num_elements, size, rng);
  i32 dim = (sync_type == gplda::block) ? blockDim.x : warpSize;
  i32 thread_idx = threadIdx.x % dim;

  // insert elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    u32 i = offset * dim + thread_idx;
    m->insert(i, i < num_elements ? i : 0);
  }

  // sync if needed
  if(sync_type == gplda::block) {
    __syncthreads();
  }

  // optionally, rebuild
  if(rebuild == true) {
    m->rebuild(0);
  }

  // retrieve elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    i32 i = offset * dim + thread_idx;
    if(i < num_elements) {
      out[i] = m->get(i);
    }
  }
}



void test_hash_map() {
  constexpr u32 size = 10000;
  constexpr u32 num_elements = 9000; // large contention to ensure stash is used
  constexpr u32 map_size = 2 * (size + GPLDA_HASH_STASH_SIZE);

  curandStatePhilox4_32_10_t* rng;
  cudaMalloc(&rng, sizeof(curandStatePhilox4_32_10_t)) >> GPLDA_CHECK;
  gplda::rng_init<<<1,1>>>(0,0,rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  void* map;
  cudaMalloc(&map, map_size * sizeof(u64)) >> GPLDA_CHECK;
  u64* map_host = new u64[map_size];

  u32* out;
  cudaMalloc(&out, num_elements * sizeof(u32)) >> GPLDA_CHECK;
  u32* out_host = new u32[num_elements];

  // init<warp>
  test_hash_map_init<gplda::warp><<<1,32>>>(map, size, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(map_host, map, map_size * sizeof(u64), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < size + GPLDA_HASH_STASH_SIZE; ++i) {
    assert(map_host[i] == GPLDA_HASH_EMPTY);
    map_host[i] = 0;
  }

  // init<block>
  test_hash_map_init<gplda::block><<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, size, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(map_host, map, map_size * sizeof(u64), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < size + GPLDA_HASH_STASH_SIZE; ++i) {
    assert(map_host[i] == GPLDA_HASH_EMPTY);
    map_host[i] = 0;
  }

  // insert<warp, no_rebuild>
  test_hash_map_insert<gplda::warp, false><<<1,32>>>(map, size, num_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_elements; ++i) {
    assert(out_host[i] == i);
    out_host[i] = 0;
  }

  // insert<block, no_rebuild>
  test_hash_map_insert<gplda::block, false><<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, size, num_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_elements; ++i) {
    assert(out_host[i] == i);
    out_host[i] = 0;
  }

  // insert<warp, rebuild>
  test_hash_map_insert<gplda::warp, true><<<1,32>>>(map, size, num_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_elements; ++i) {
    assert(out_host[i] == i);
    out_host[i] = 0;
  }

  // insert<block, rebuild>
  test_hash_map_insert<gplda::block, true><<<1,GPLDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(map, size, num_elements, out, rng);
  cudaDeviceSynchronize() >> GPLDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPLDA_CHECK;

  for(i32 i = 0; i < num_elements; ++i) {
    assert(out_host[i] == i);
    out_host[i] = 0;
  }




  // cleanup
  cudaFree(out);
  delete[] out_host;
  cudaFree(map);
  delete[] map_host;
  cudaFree(rng);
}


}
