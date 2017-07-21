#ifndef GPLDA_POISSON_H
#define GPLDA_POISSON_H

#include "stdint.h"
#include "spalias.cuh"

namespace gplda {

class Poisson {
  public:
    SpAlias* alias;
    int max_lambda;
    int max_value;
    Poisson(int ml, int mv);
    ~Poisson();
};

}

#endif
