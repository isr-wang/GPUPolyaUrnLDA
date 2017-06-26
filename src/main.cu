/*
 ============================================================================
 Name        : GPUPolyaUrnLDA.cu
 Author      : 
 Version     :
 Copyright   : 
 Description :
 ============================================================================
 */

#include <stdlib.h>

void printUsage() {
  std::cerr
    << "usage: gplda <input> <output>"
    << std::endl;
}

int main(int argc, char** argv) {
	if(argc != 2) {
		printUsage();
		exit(EXIT_FAILURE);
	}
	GPLDA(argc, argv);
	return 0;
}
