#include <iostream>
// #include <intrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <cfloat>
#include <vector>
#include "sse_l2_space.h"
#include "pq_dis.h"
#include "sse_test.h"
#include "avx2_l2_space.h"

int main() {
  test_sse_l2_space();
  test_avx2_l2_space();
  test_pq_dis();
  // sse_test();
  return 0;
}