#include <immintrin.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <cstring>

constexpr size_t dim = 128;
constexpr size_t num_vectors = 100000;
constexpr size_t pq_m = 8;    // subspaces
constexpr size_t pq_k = 256;  // clusters
constexpr size_t subspace_dim = dim / pq_m;

// 用于存储 LUT：8 个子空间，每个 256 个中心点，存 float 距离
float pq_lut[pq_m][pq_k];

// 每个数据库向量被编码成 8 个 uint8_t
std::vector<std::vector<uint8_t>> pq_codes;

// 原始向量用于 AVX 比较
std::vector<std::vector<float>> raw_vectors;

// 查询向量（随机）
float query[dim];

// AVX2 内积
float avx_inner_product(const float* a, const float* b) {
  __m256 sum = _mm256_setzero_ps();
  for (int i = 0; i < dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
  }
  float result[8];
  _mm256_storeu_ps(result, sum);
  float total = 0.0f;
  for (int i = 0; i < 8; i++) total += result[i];
  return total;
}

// PQ 查表距离
float pq_lookup_distance(const uint8_t* code) {
  float sum = 0.0f;
  for (int m = 0; m < pq_m; ++m) {
    sum += pq_lut[m][code[m]];
  }
  return sum;
}

int main() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> code_dist(0, pq_k - 1);

  // 生成 PQ LUT
  for (int m = 0; m < pq_m; ++m)
    for (int k = 0; k < pq_k; ++k) pq_lut[m][k] = dist(rng);

  // 生成 PQ 编码
  pq_codes.resize(num_vectors, std::vector<uint8_t>(pq_m));
  for (auto& code : pq_codes)
    for (auto& c : code) c = code_dist(rng);

  // 生成原始向量
  raw_vectors.resize(num_vectors, std::vector<float>(dim));
  for (auto& vec : raw_vectors)
    for (auto& val : vec) val = dist(rng);

  // 查询向量
  for (auto& val : query) val = dist(rng);

  // PQ 查表计时
  auto t1 = std::chrono::high_resolution_clock::now();
  float pq_sum = 0.0f;
  for (size_t i = 0; i < num_vectors; ++i) {
    pq_sum += pq_lookup_distance(pq_codes[i].data());
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> pq_time = t2 - t1;

  // AVX 计算计时
  float avx_sum = 0.0f;
  t1 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_vectors; ++i) {
    avx_sum += avx_inner_product(query, raw_vectors[i].data());
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> avx_time = t2 - t1;

  std::cout << "PQ total:  " << pq_sum << ", time = " << pq_time.count() << " s\n";
  std::cout << "AVX total: " << avx_sum << ", time = " << avx_time.count() << " s\n";

  return 0;
}
