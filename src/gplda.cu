
#include <stdlib.h>

void GPLDA(int argc, char** argv) {
	// parse arguments and set configuration
	std::string inFile(argv[1]);
	std::string outFile(argv[2]);
	
	preprocess(inFile);
	train(500);
	output(outFile);
}
