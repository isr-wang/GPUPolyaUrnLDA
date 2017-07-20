#ifndef GPLDA_SPALIAS_H
#define GPLDA_SPALIAS_H

namespace gplda {

class SpAlias {
  public:
    SpAlias();
    ~SpAlias();
};

__global__ void build_alias(float** prob, float** alias, int table_size);

}

#endif
