/*
 ============================================================================
 Name        : main.cu
 Author      : 
 Version     :
 Copyright   : 
 Description :
 ============================================================================
 */

#include <stdlib.h>
#include "args.h"

namespace gplda {

int main(int argc, char** argv) {
  args::parse(argc,argv);
  preprocess();
  train();
  output();
  return 0;
}

}
