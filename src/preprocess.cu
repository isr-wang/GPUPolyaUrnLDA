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
  remove(args::zTempFile.c_str()); remove(args::wTempFile.c_str()); remove(args::dTempFile.c_str());

  std::string line;

  std::ifstream inputStream(args::input);
  std::ofstream zStream(args::zTempFile, std::ios::binary);
  std::ofstream wStream(args::wTempFile, std::ios::binary);
  std::ofstream dStream(args::dTempFile, std::ios::binary);

  if(!inputStream.is_open()) { std::cout << "Input file not found"; exit(EXIT_FAILURE); }
  if(!zStream.is_open()) { std::cout << "Could not create z temp file"; exit(EXIT_FAILURE); }
  if(!wStream.is_open()) { std::cout << "Could not create w temp file"; exit(EXIT_FAILURE); }
  if(!dStream.is_open()) { std::cout << "Could not create d temp file"; exit(EXIT_FAILURE); }

  std::minstd_rand rand(args::seed);
  std::uniform_int_distribution<> unif(0,args::K);

  int d = 0;
  while(std::getline(inputStream, line)) {
    std::vector<int> tokens = getTokens(line);

    for(int i = 0; i < tokens.size(); ++i) { int r = unif(rand); zStream.write(reinterpret_cast<char*>(&r), sizeof(r)); }
    wStream.write(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(int));
    dStream.write(reinterpret_cast<char*>(&d), sizeof(d));
    d++;
  }

  inputStream.close();
  zStream.close();
  wStream.close();
  dStream.close();

}

}
