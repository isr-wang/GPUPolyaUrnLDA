#include "hashmap.cuh"
#include "warpsample.cuh"
#include <thrust/system/cuda/detail/cub/block/block_scan.cuh>
#include <thrust/system/cuda/detail/cub/warp/warp_scan.cuh>

namespace gplda {

__global__ void compute_d_idx(uint32_t* d_len, uint32_t* d_idx, uint32_t n_docs) {
  typedef cub::BlockScan<int32_t, GPLDA_COMPUTE_D_IDX_BLOCKDIM> BlockScan;
  __shared__ typename BlockScan::TempStorage temp;

  if(blockIdx.x == 0) {
    int32_t thread_d;
    int32_t initial_value = 0;
    int32_t total_value;
    for(int32_t offset = 0; offset < n_docs / blockDim.x + 1; ++offset) {
      int32_t i = threadIdx.x + offset * blockDim.x;
      if(i < n_docs) {
        thread_d = d_len[i];
      } else {
        thread_d = 0;
      }

      BlockScan(temp).ExclusiveScan(thread_d, thread_d, 0, cub::Sum(), total_value);

      // workaround for CUB bug: apply offset manually
      __syncthreads();
      thread_d = thread_d + initial_value;
      initial_value = total_value + initial_value;

      if(i < n_docs) {
        d_idx[i] = thread_d;
      }
    }
  }
}

__device__ __forceinline__ uint32_t draw_alias(float u, float* prob, uint32_t* alias, uint32_t table_size, int32_t lane_idx) {
  uint32_t ret = 0;
  if(lane_idx == 0) {
    // determine the slot and update random number
    float ts = (float) table_size;
    uint32_t slot = (uint32_t) (u * ts);
    u = fmodf(u, __frcp_rz(ts)) * ts;

    // load table elements from global memory
    float thread_prob = prob[slot];
    uint32_t thread_alias = alias[slot];

    // return the resulting draw
    if(u < thread_prob) {
      ret = slot;
    } else {
      ret = thread_alias;
    }
  }
  return ret;
}

__device__ __forceinline__ uint32_t draw_wary_search(float u) {
  return 0;
}

__device__ __forceinline__ void count_topics(uint32_t* z, uint32_t document_size, HashMap* m, void* temp, int32_t lane_idx, curandStatePhilox4_32_10_t* rng) {
  // initialize the hash table
  hash_map_init(m, temp, document_size, warpSize, rng);

  // loop over z, add to m
  for(int32_t offset = 0; offset < document_size / warpSize + 1; ++offset) {
    int32_t i = offset * warpSize + lane_idx;
    if(i < document_size) {
      hash_map_accumulate(z[i], i, m);
    }
  }
}

__device__ __forceinline__ float compute_product_cumsum(uint32_t* mPhi, HashMap* m, float* Phi_dense, int32_t warp_idx, cub::WarpScan<int32_t>::TempStorage* temp) {
  int32_t thread_mPhi = 0;
  cub::WarpScan<int32_t>(temp[warp_idx]).ExclusiveSum(thread_mPhi, thread_mPhi);
  return 0.0f;
}

__global__ void warp_sample_topics(uint32_t size, uint32_t n_docs,
    uint32_t* z, uint32_t* w, uint32_t* d_len, uint32_t* d_idx, uint32_t* K_d, void* temp,
    uint32_t K, uint32_t V, uint32_t max_K_d,
    float* Phi_dense,
    float** prob, uint32_t** alias, curandStatePhilox4_32_10_t* rng) {
  // initialize variables
  int32_t lane_idx = threadIdx.x % warpSize;
  int32_t warp_idx = threadIdx.x / warpSize;
  curandStatePhilox4_32_10_t warp_rng = rng[0];
  HashMap m;
  uint32_t** mPhi = (uint32_t**) &m.temp_data;
  __shared__ typename cub::WarpScan<int32_t>::TempStorage warp_scan_temp[1];

  // loop over documents
  for(int32_t i = 0; i < n_docs; ++i) {
    // count topics in document
    uint32_t warp_d_len = d_len[i];
    uint32_t warp_d_idx = d_idx[i];
    count_topics(z + warp_d_idx * sizeof(uint32_t), warp_d_len, &m, temp, lane_idx, &warp_rng);

    // loop over words
    for(int32_t j = 0; j < warp_d_len; ++j) {
      // load z,w from global memory
      uint32_t warp_z = z[warp_d_idx + j];
      uint32_t warp_w = 0;//w[warp_d_idx + j]; // why is this broken?

      // remove current z from sufficient statistic
      hash_map_accumulate(warp_z, lane_idx == 0 ? -1 : 0, &m); // decrement on 1st lane without branching

      // compute m*phi and sigma_b
      float warp_sigma_a = 0.0f;
      float sigma_b = compute_product_cumsum(*mPhi, &m, Phi_dense, warp_idx, warp_scan_temp);

      // update z
      float u1 = curand_uniform(&warp_rng);
      float u2 = curand_uniform(&warp_rng);
      if(u1 * (warp_sigma_a + sigma_b) > warp_sigma_a) {
        // sample from m*Phi
        warp_z = draw_wary_search(u2);
      } else {
        // sample from alias table
        warp_z = draw_alias(u2, prob[warp_w], alias[warp_w], /*table_size =*/ 1, lane_idx);
      }

      // add new z to sufficient statistic
      hash_map_accumulate(warp_z, lane_idx == 0, &m); // increment on 1st lane without branching
      if(lane_idx == 0) {
        z[warp_d_idx + j] = warp_z;
      }
    }
  }
}

}
