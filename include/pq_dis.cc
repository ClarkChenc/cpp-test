#include <iostream>
#include <chrono>
#include <unordered_map>
#include <vector>

#include "utils.h"

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

    uint16_t ret1 = 0;
    uint16_t ret2 = 0;
    for (size_t j = 0; j < subspace_num; j += 2) {
      ret1 += matrix_ptr[j * cluster_num + cur_encode[j]];
      ret2 += matrix_ptr[(j + 1) * cluster_num + cur_encode[j + 1]];
    }
    res[i] = ret1 + ret2;
  }

  auto e_search = std::chrono::steady_clock::now();

  uint64_t sum = 0;
  for (int i = 0; i < query_num; ++i) {
    sum += res[i];
  }
  std::cout << "sum: " << sum << std::endl;
  std::cout << "cost: " << time_cost(s_search, e_search) / 1000000 << " ms" << std::endl;
}