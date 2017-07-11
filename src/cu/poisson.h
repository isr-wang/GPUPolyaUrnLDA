#ifndef GPLDA_POISSON_H
#define GPLDA_POISSON_H

#include "stdint.h"

namespace gplda {

class Poisson {
  public:
    float** prob;
    float** alias;
    Poisson(uint32_t lambda_max, size_t size);
    ~Poisson();
};

__global__ void build_poisson(float** prob, float** alias, float beta, uint32_t lambda, size_t size);

}

#endif
