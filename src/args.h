#include <string>

namespace gplda {

namespace args {
  extern float alpha;
  extern float beta;
  extern unsigned int K;
  extern unsigned int nMC;
  extern long seed;
  extern int bufferSize;
  extern std::string input;
  extern std::string output;
  extern std::string zTempFile;
  extern std::string wTempFile;
  extern std::string dTempFile;
  void parse(int, char**);
  void printUsage();
}

}
