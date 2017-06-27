#include <stdlib.h>
#include <string.h>
#include <iostream>

namespace gplda {

namespace args {
  float alpha = 0.1;
  float beta = 0.1;
  unsigned int K = 10;
  unsigned int nMC = 100;
  unsigned long seed = 0;
  int bufferSize = 1024;
  std::string input = "data/small.txt";
  std::string output = "output/small.txt";
  std::string zTempFile = "temp/z.bin";
  std::string wTempFile = "temp/w.bin";
  std::string dTempFile = "temp/d.bin";

  void printUsage() {
    std::cerr
      << "usage: gplda <asdf>"
      << std::endl;
  }

  void parse(int argc, char** argv) {
    std::cout << "parsing args" << std::endl;
  }
}

}


///**
// * Copyright © 2017 Kunal Sarkhel <ksarkhel@gmu.edu>
// *
// * META: I suppose we should figure out a place to put an AUTHORS file and
// * subsequently standardize the copyright header for these file.
// */
//
//#include <functional>
//#include <iostream>
//#include <sys/stat.h>
//#include <unistd.h>
//#include <unordered_map>
//#include "args.hxx"
//#include "LDA.hxx"
//
///* TODO: This version string should come from Git and the pre-processor */
//#define PROG_VERSION "0.0.1"
//
//#define xstr(s) str(s)
//#define str(s) #s
//
//#define DEFAULT_NUM_TOPICS 10
//#define DEFAULT_NUM_ITERATIONS 1000
//#define DEFAULT_ALPHA 5
//#define DEFAULT_BETA 0.01
//
//inline bool file_exists(const std::string& filename) {
//  struct stat buffer;
//  return (stat(filename.c_str(), &buffer) == 0);
//}
//
//void ImportFile(const std::string& progname,
//                std::vector<std::string>::const_iterator beginargs,
//                std::vector<std::string>::const_iterator endargs);
//void TrainTopics(const std::string& progname,
//                 std::vector<std::string>::const_iterator beginargs,
//                 std::vector<std::string>::const_iterator endargs);
//
//using commandtype = std::function<void(
//    const std::string&, std::vector<std::string>::const_iterator,
//    std::vector<std::string>::const_iterator)>;
//
//int main(int argc, char* argv[]) {
//  /**
//   * TODO: CUDA 8 supposedly fixes the unordered_map bug, but it still causes a
//   * compilation error for .cu files. Figure out an alternative or make sure
//   * the CLI is a CPP file and the kernels and CUDA code live in a .cu file.
//   */
//  std::unordered_map<std::string, commandtype> map{
//      {"import-file", ImportFile}, {"train-topics", TrainTopics}};
//
//  const std::vector<std::string> args(argv + 1, argv + argc);
//  args::ArgumentParser parser(
//      "A GPU implementation of the Pólya Urn LDA sampler for topic modeling",
//      "Valid commands are import-file and train-topics");
//  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
//  parser.Prog(argv[0]);
//  args::Flag version(parser, "version", "Show the version of this program",
//                     {"version"});
//  args::MapPositional<std::string, commandtype> command(
//      parser, "command", "Command to execute", map);
//  command.KickOut(true);
//
//  /**
//   * NOTE: I can re-write this so that it does not exceptions if necessary.
//   */
//  try {
//    auto next = parser.ParseArgs(args);
//    std::cout << std::boolalpha;
//
//    if (bool{version}) {
//      std::cout << argv[0] << " v" << PROG_VERSION << std::endl;
//      return 0;
//    }
//    if (command) {
//      args::get(command)(argv[0], next, std::end(args));
//    } else {
//      std::cout << parser;
//    }
//  } catch (args::Help) {
//    std::cout << parser;
//    return 0;
//  } catch (args::Error e) {
//    std::cerr << e.what() << std::endl;
//    std::cerr << parser;
//    return 1;
//  }
//  return 0;
//}
//
//void ImportFile(const std::string& progname,
//                std::vector<std::string>::const_iterator beginargs,
//                std::vector<std::string>::const_iterator endargs) {
//  args::ArgumentParser parser(
//      "Convert an input file containing one document per line to a binary file "
//      "for training");
//  parser.Prog(progname + " import-file");
//  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
//  args::Flag remove_stopwords(
//      parser, "remove stopwords",
//      "Remove common adverbs, conjunctions, prepositions, pronouns and such",
//      {"remove-stopwords"});
//  args::ValueFlag<std::string> input(
//      parser, "FILE",
//      "Input file containing one document per line with space separated tokens",
//      {'i', "input"});
//  args::ValueFlag<std::string> output(
//      parser, "FILE",
//      "Binary output file consisting of topic indicator and tokens",
//      {'o', "output"});
//
//  try {
//    parser.ParseArgs(beginargs, endargs);
//    if (!bool{input} || !bool{output}) {
//      std::cerr << "Both input and output flags are required" << std::endl;
//      std::cout << parser;
//      // TODO: Figure out how to exit with a non-zero errorcode here.
//      return;
//    }
//    if (!file_exists(args::get(input))) {
//      std::cout << "Input file " << args::get(input)
//                << " does not exist. Exiting immediately." << std::endl;
//      return;
//    }
//    std::cout << "Input file: " << args::get(input) << std::endl;
//    if (file_exists(args::get(output))) {
//      std::cout << "Output file " << args::get(output)
//                << " already exists and will be overwritten!" << std::endl;
//    } else {
//      std::cout << "Output file: " << args::get(output) << std::endl;
//    }
//    // TODO: Perhaps check if the input and output file are the same
//
//    convertFile(args::get(input), args::get(output));
//
//  } catch (args::Help) {
//    std::cout << parser;
//    return;
//  } catch (args::ParseError e) {
//    std::cerr << e.what() << std::endl;
//    std::cerr << parser;
//    return;
//  }
//}
//
//void TrainTopics(const std::string& progname,
//                 std::vector<std::string>::const_iterator beginargs,
//                 std::vector<std::string>::const_iterator endargs) {
//  args::ArgumentParser parser("");
//  parser.Prog(progname + " train-topics");
//  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
//  args::ValueFlag<std::string> input(
//      parser, "FILE",
//      "Binary input file consisting of topic indicators and tokens",
//      {'i', "input"});
//  args::ValueFlag<std::string> output_state(
//      parser, "FILE", "LDA model state file in Mallet format",
//      {"output-state"});
//  args::ValueFlag<std::string> output_model(parser, "FILE", "LDA model file",
//                                            {"output-model"});
//  args::ValueFlag<int> num_topics(
//      parser, "INTEGER",
//      "Number of topics to fit. Default is " xstr(DEFAULT_NUM_TOPICS),
//      {"num-topics"});
//  args::ValueFlag<int> num_iterations(
//      parser, "INTEGER",
//      "Number of iterations to fit. Default is " xstr(DEFAULT_NUM_ITERATIONS),
//      {"num-iterations"});
//  args::ValueFlag<int> random_seed(
//      parser, "INTEGER",
//      "Random seed for the sampler. Defaults to 0 (which uses the clock). ",
//      {"random-seed"});
//  args::ValueFlag<double> alpha(
//      parser, "DECIMAL",
//      "Sum over topics of smoothing over document-topic distributions; alpha_k "
//      "= alpha / [num topics]. Default is " xstr(DEFAULT_ALPHA),
//      {"alpha"});
//  args::ValueFlag<double> beta(
//      parser, "DECIMAL",
//      "Smoothing parameter for each topic-word; beta_w. Default is " xstr(
//          DEFAULT_BETA),
//      {"beta"});
//
//  try {
//    parser.ParseArgs(beginargs, endargs);
//    if (!bool{input}) {
//      std::cerr << "Input file is required" << std::endl;
//      std::cout << parser;
//      // TODO: Figure out how to exit with a non-zero errorcode here.
//      return;
//    }
//    if (!file_exists(args::get(input))) {
//      std::cout << "Input file " << args::get(input)
//                << " does not exist. Exiting immediately." << std::endl;
//      return;
//    }
//    std::cout << "Input file: " << args::get(input) << std::endl;
//    if (file_exists(args::get(output_model))) {
//      std::cout << "Output model file " << args::get(output_model)
//                << " already exists and will be overwritten!" << std::endl;
//    } else {
//      std::cout << "Output model file: " << args::get(output_model)
//                << std::endl;
//    }
//    if (file_exists(args::get(output_state))) {
//      std::cout << "Output state file " << args::get(output_state)
//                << " already exists and will be overwritten!" << std::endl;
//    } else {
//      std::cout << "Output state file: " << args::get(output_state)
//                << std::endl;
//    }
//    // TODO: Perhaps check if the input and output files are the same
//
//    int lda_num_topics =
//        bool{num_topics} ? args::get(num_topics) : DEFAULT_NUM_TOPICS;
//    int lda_num_iterations = bool{num_iterations} ? args::get(num_iterations)
//                                                  : DEFAULT_NUM_ITERATIONS;
//    double lda_alpha = bool{alpha} ? args::get(alpha) : DEFAULT_ALPHA;
//    double lda_beta = bool{beta} ? args::get(beta) : DEFAULT_BETA;
//    int lda_random_seed = bool{random_seed} ? args::get(random_seed) : 0;
//
//    std::cout << "num_topics: " << lda_num_topics << std::endl;
//    std::cout << "num_iterations: " << lda_num_iterations << std::endl;
//    std::cout << "alpha: " << lda_alpha << std::endl;
//    std::cout << "beta: " << lda_beta << std::endl;
//    if (lda_random_seed == 0) {
//      std::cout << "random_seed: 0 (use clock)" << std::endl;
//    } else {
//      std::cout << "random_seed: " << lda_random_seed << std::endl;
//    }
//
//    LDA(args::get(input), args::get(output_model), args::get(output_state),
//        lda_num_topics, lda_num_iterations, lda_alpha, lda_beta,
//        lda_random_seed);
//
//  } catch (args::Help) {
//    std::cout << parser;
//    return;
//  } catch (args::ParseError e) {
//    std::cerr << e.what() << std::endl;
//    std::cerr << parser;
//    return;
//  }
//}
//
