#ifndef GPLDA_SPALIAS_H
#define GPLDA_SPALIAS_H

#include "stdint.h"

namespace gplda {

class SpAlias {
  public:
    uint32_t num_tables;
    uint32_t table_size;
    float** prob;
    uint32_t** alias;
    SpAlias(uint32_t nt, uint32_t ts);
    ~SpAlias();
};

__host__ __device__ uint32_t next_pow2(uint32_t x);

__global__ void build_alias(float** prob, uint32_t** alias, uint32_t table_size);

}

#endif
