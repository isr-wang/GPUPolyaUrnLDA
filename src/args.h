namespace gplda {

class Args {
  public:
    Args();
    float alpha;
    float beta;
    unsigned int K;
    unsigned int nMC;
    long seed;
    void parse(int, char**);
    void printUsage();
};

}
