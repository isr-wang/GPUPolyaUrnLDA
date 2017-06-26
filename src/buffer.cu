#include <iostream>
#include <stdlib.h>
#include "args.h"

namespace gplda {

Buffer::Buffer() {
  z = new unsigned int[args::bufferSize];
  w = new unsigned short[args::bufferSize];
  dLen = new int[args::bufferSize];
  dIdx = new int[args::bufferSize];
  nDocs = 0;
}

Buffer::~Buffer() {
  delete[] z;
  delete[] w;
  delete[] dLen;
  delete[] dIdx;
  delete nDocs;
}

}
