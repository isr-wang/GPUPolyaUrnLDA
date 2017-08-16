#pragma once

#include "types.cuh"
#include "spalias.cuh"

namespace gplda {

class Poisson {
  public:
    SpAlias* pois_alias;
    u32 max_lambda;
    u32 max_value;
    f32 beta;
    Poisson(u32 ml, u32 mv, f32 b);
    ~Poisson();
};

}
