namespace gplda {

class Buffer {
  public:
    Buffer();
    ~Buffer();
    unsigned int *z;
    unsigned int *w;
    unsigned int *dIdx;
    unsigned int *dLen;
    unsigned int nDocs;
};

}
