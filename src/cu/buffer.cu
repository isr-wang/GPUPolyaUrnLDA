#include <iostream>
#include <stdlib.h>
#include "args.h"
#include "buffer.h"

namespace gplda {

Buffer::Buffer() {
  z = new unsigned int[args::bufferSize];
  w = new unsigned int[args::bufferSize];
  dLen = new unsigned int[args::bufferSize];
  dIdx = new unsigned int[args::bufferSize];
  nDocs = 0;
}

Buffer::~Buffer() {
  delete[] z;
  delete[] w;
  delete[] dLen;
  delete[] dIdx;
}

}
