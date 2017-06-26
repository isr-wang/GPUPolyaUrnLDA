#include <iostream>
#include <string>
#include "LDA.hxx"

void convertFile(std::string input_filename, std::string output_filename) {
  std::cout << "Converting " << input_filename << " to optimized binary format"
            << std::endl;
  std::cout << "Optimized file will be written to " << output_filename;
}

void LDA(std::string input_filename, std::string output_model,
         std::string output_state, int num_topics, int num_iterations,
         double alpha, double beta, int random_seed) {
  return;
}
