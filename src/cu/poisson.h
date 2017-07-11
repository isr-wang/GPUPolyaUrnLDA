#ifndef GPLDA_POISSON_H
#define GPLDA_POISSON_H

#include "stdint.h"

namespace gplda {

class Poisson {
  public:
    uint32_t max_lambda;
    size_t max_value;
    float** prob;
    float** alias;
    Poisson(uint32_t lambda_max, size_t size);
    ~Poisson();
};

}

#endif
