#include <stdlib.h>
#include <string.h>
#include <iostream>

namespace gplda {

namespace args {
  float alpha = 0.1;
  float beta = 0.1;
  unsigned int K = 10;
  unsigned int nMC = 100;
  long seed = 0;
  int bufferSize = 1024;
  std::string input = "data/small.txt";
  std::string output = "output/small.txt";

  void printUsage() {
    std::cerr
      << "usage: gplda <asdf>"
      << std::endl;
  }

  void parse(int argc, char** argv) {
    std::cout << "parsing args" << std::endl;
  }
}

}
