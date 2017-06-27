#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "args.h"

namespace gplda {

inline std::vector<int> getTokens(const std::string& line) {
  std::vector<int> tokens;
  tokens.push_back(1);
  return tokens;
}

void preprocess() {
  std::cout << "preprocessing" << std::endl;
  std::string line;

  std::ifstream inputStream(args::input);
  std::ofstream zStream(args::zTempFile, std::ios::binary);
  std::ofstream wStream(args::wTempFile, std::ios::binary);
  std::ofstream dStream(args::dTempFile, std::ios::binary);

  if(!inputStream.is_open()) { throw "Input file not found"; }
  if(!zStream.is_open()) { throw "Could not create z temp file"; }
  if(!wStream.is_open()) { throw "Could not create w temp file"; }
  if(!dStream.is_open()) { throw "Could not create d temp file"; }

  std::minstd_rand rand(args::seed);
  std::uniform_int_distribution<> unif(0,args::K);

  unsigned long d = 0;
  while(std::getline(inputStream, line)) {
    std::vector<int> tokens = getTokens(line);

    for(int i = 0; i < tokens.size(); ++i) zStream << unif(rand);
    wStream.write(reinterpret_cast<char*>(tokens.data()), tokens.size());
    dStream << d;
    d++;
  }

}

}
