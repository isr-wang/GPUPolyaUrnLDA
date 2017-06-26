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
  Args args = Args();
  args.parse(argc,argv);
  preprocess(args);
  train(args);
  output(args);
  return 0;
}

}
