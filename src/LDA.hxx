#ifndef LDA_H_
#define LDA_H_

#include <string>

void convertFile(std::string input_filename, std::string output_filename);

void LDA(std::string input_filename, std::string output_model,
         std::string output_state, int num_topics, int num_iterations,
         double alpha, double beta, int random_seed);

#endif /* LDA_H_ */
