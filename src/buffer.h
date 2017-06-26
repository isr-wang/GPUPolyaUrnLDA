namespace gplda {

class Buffer {
  public:
    Buffer();
    ~Buffer();
    unsigned int[] z;
    unsigned short[] w;
    int[] dIdx;
    int[] dLen;
    int nDocs;
};

}
