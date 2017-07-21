#ifndef GPLDA_SPALIAS_H
#define GPLDA_SPALIAS_H

namespace gplda {

class SpAlias {
  public:
    int num_tables;
    int table_size;
    float** prob;
    float** alias;
    SpAlias(int nt, int ts);
    ~SpAlias();
};

__host__ __device__ unsigned int next_pow2(unsigned int x);

__global__ void build_alias(float** prob, float** alias, int table_size);

}

#endif
