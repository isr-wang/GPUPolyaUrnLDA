#include <stdlib.h>
#include <iostream>
#include "test_polyaurn.cuh"
#include "test_spalias.cuh"
#include "test_warpsample.cuh"

namespace gplda_test {

void run_tests() {
  std::cout << "running tests" << std::endl;

  std::cout << "testing polya_urn_init" << std::endl;
  test_polya_urn_init();
  std::cout << "testing polya_urn_sample" << std::endl;
  test_polya_urn_sample();
  std::cout << "testing polya_urn_transpose" << std::endl;
  test_polya_urn_transpose();
  std::cout << "testing polya_urn_colsums" << std::endl;
  test_polya_urn_colsums();

  std::cout << "testing build_alias" << std::endl;
  test_build_alias();

  std::cout << "testing compute_d_idx" << std::endl;
  test_compute_d_idx();
  std::cout << "testing warp_sample_topics" << std::endl;
  test_warp_sample_topics();

  std::cout << "tests completed" << std::endl;
}

}
