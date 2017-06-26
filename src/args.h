namespace gplda {

namespace args {
  extern float alpha;
  extern float beta;
  extern unsigned int K;
  extern unsigned int nMC;
  extern long seed;
  int bufferSize;
  extern std::string input;
  extern std::string output;
  void parse(int, char**);
  void printUsage();
}

}
