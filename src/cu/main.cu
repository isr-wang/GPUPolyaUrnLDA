/*
 ============================================================================
 Name        : main.cu
 Author      : 
 Version     :
 Copyright   : 
 Description :
 ============================================================================
 */

#include <iostream>
#include "args.h"
#include "preprocess.h"
#include "train.h"
#include "output.h"

using namespace gplda;

int main(int argc, char** argv) {
  args::parse(argc,argv);
  preprocess();
  train();
  output();
  return 0;
}
