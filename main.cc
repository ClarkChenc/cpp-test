#include <iostream>
// #include <intrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <cfloat>
#include <vector>
#include "include.h"

int main() {
  test_sse_l2_space();
  test_avx2_l2_space();
  test_pq_dis();
  // sse_test();
  return 0;
}