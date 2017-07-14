#ifndef GPLDA_POISSON_H
#define GPLDA_POISSON_H

#include "stdint.h"

namespace gplda {

class Poisson {
  public:
    size_t max_lambda;
    size_t max_value;
    float** prob;
    float** alias;
    Poisson(size_t ml, size_t mv);
    ~Poisson();
};

}

#endif
