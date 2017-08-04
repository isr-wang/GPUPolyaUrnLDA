#ifndef GPLDA_POISSON_H
#define GPLDA_POISSON_H

#include "stdint.h"
#include "spalias.cuh"

namespace gplda {

class Poisson {
  public:
    SpAlias* pois_alias;
    uint32_t max_lambda;
    uint32_t max_value;
    float beta;
    Poisson(uint32_t ml, uint32_t mv, float b);
    ~Poisson();
};

}

#endif
