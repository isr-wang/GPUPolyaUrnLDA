#ifndef GPLDA_POISSON_H
#define GPLDA_POISSON_H

#include "stdint.h"

namespace gplda {

class Poisson {
  public:
    int max_lambda;
    int max_value;
    float** prob;
    float** alias;
    Poisson(int ml, int mv);
    ~Poisson();
};

}

#endif
