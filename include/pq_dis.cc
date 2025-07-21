#include <iostream>
#include <chrono>
#include <unordered_map>
#include <vector>

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>  // SSE3 指令集

#include "utils.h"

int horizontal_add_epi32(__m128i v) {
  __m128i sum = _mm_hadd_epi32(v, v);  // [x0+x1, x2+x3, x0+x1, x2+x3]
  sum = _mm_hadd_epi32(sum, sum);      // [x0+x1+x2+x3, ...]
  return _mm_cvtsi128_si32(sum);
}

uint16_t get_pq_dis(const void* p_vec1, const void* p_vec2, size_t subspace_num, size_t cluster_num) {
  uint16_t dis = 0;

  uint16_t* ptr_vec1 = (uint16_t*)p_vec1;
  uint8_t* ptr_vec2 = (uint8_t*)p_vec2;

  __m128i sum = _mm_setzero_si128();
  __m128i v1;
  __m128i v2;
  __m128i tmp;
  for (size_t i = 0; i < subspace_num; i += 8) {
    v1 = _mm_set_epi32(ptr_vec1[ptr_vec2[0]], ptr_vec1[1 * cluster_num + ptr_vec2[1]],
                       ptr_vec1[2 * cluster_num + ptr_vec2[2]], ptr_vec1[3 * cluster_num + ptr_vec2[3]]);
    v2 = _mm_set_epi32(ptr_vec1[4 * cluster_num + ptr_vec2[4]], ptr_vec1[5 * cluster_num + ptr_vec2[5]],
                       ptr_vec1[6 * cluster_num + ptr_vec2[6]], ptr_vec1[7 * cluster_num + ptr_vec2[7]]);

    tmp = _mm_add_epi32(v1, v2);
    sum = _mm_add_epi32(sum, tmp);
    ptr_vec1 += 8 * cluster_num;
    ptr_vec2 += 8;
  }
  dis = horizontal_add_epi32(sum);

  return dis;
}
uint16_t get_pq_dis_plane(const void* p_vec1, const void* p_vec2, size_t subspace_num, size_t cluster_num) {
  uint16_t* matrix_ptr = (uint16_t*)p_vec1;
  uint8_t* cur_encode = (uint8_t*)p_vec2;

  uint16_t ret1 = 0;
  uint16_t ret2 = 0;
  for (size_t j = 0; j < subspace_num; j += 2) {
    ret1 += matrix_ptr[j * cluster_num + cur_encode[j]];
    ret2 += matrix_ptr[(j + 1) * cluster_num + cur_encode[j + 1]];
  }
  return ret1 + ret2;
}

void test_pq_dis() {
  size_t subspace_num = 128;
  size_t cluster_num = 64;
  size_t query_num = 100;
  query_num *= 10000;

  uint16_t* matrix = (uint16_t*)malloc(subspace_num * cluster_num * sizeof(int16_t));

  for (size_t i = 0; i < subspace_num * cluster_num; ++i) {
    matrix[i] = i;
  }

  // uint8_t* encodes = (uint8_t*)malloc(query_num * subspace_num * sizeof(uint8_t));
  // for (size_t i = 0; i < query_num * subspace_num; ++i) {
  //   encodes[i] = i % 256;
  // }

  std::unordered_map<int, std::vector<uint8_t>> encode_map;
  for (size_t i = 0, k = 0; i < query_num; ++i) {
    encode_map[i] = std::vector<uint8_t>();
    for (size_t j = 0; j < subspace_num; ++j, ++k) {
      encode_map[i].push_back(k % 256);
    }
  }

  uint16_t* res = (uint16_t*)malloc(query_num * sizeof(uint16_t));
  // uint8_t* encode

  auto s_search = std::chrono::steady_clock::now();
  for (size_t i = 0; i < query_num; ++i) {
    auto* matrix_ptr = matrix;
    // auto* cur_encode = encodes + i * subspace_num;
    auto* cur_encode = encode_map[i].data();
    res[i] = get_pq_dis_plane(matrix_ptr, cur_encode, subspace_num, cluster_num);
    // res[i] = get_pq_dis(matrix_ptr, cur_encode, subspace_num, cluster_num);
  }

  auto e_search = std::chrono::steady_clock::now();

  uint64_t sum = 0;
  for (int i = 0; i < query_num; ++i) {
    sum += res[i];
  }
  std::cout << "sum: " << sum << std::endl;
  std::cout << "pq cost: " << time_cost(s_search, e_search) / 1000000 << " ms" << std::endl;
}