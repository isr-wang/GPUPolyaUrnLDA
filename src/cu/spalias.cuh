#pragma once

#include "types.cuh"

namespace gplda {

class SpAlias {
  public:
    u32 num_tables;
    u32 table_size;
    f32** prob;
    u32** alias;
    SpAlias(u32 nt, u32 ts);
    ~SpAlias();
};

__host__ __device__ u32 next_pow2(u32 x);

__global__ void build_alias(f32** prob, u32** alias, u32 table_size);

}
