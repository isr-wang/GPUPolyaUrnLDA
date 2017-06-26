#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <include/args.hxx>

namespace gplda {

Args::Args() {
  alpha = 0.1;
  beta = 0.1;
  K = 10;
  nMC = 100;
  seed = 0;
  input = "data/small.txt";
  output = "output/small.txt";
}

void Args::printUsage() {
  std::cerr
    << "usage: gplda <input> <output>"
    << std::endl;
}

void Args::parse(int argc, char** argv) {
}

}
