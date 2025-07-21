
#include <iostream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <vector>

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>

#include "utils.h"

#define HNSWLIB_avx512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))

static std::random_device rd;                                // 用于生成种子（可替换为固定种子）
static std::mt19937 gen(rd());                               // Mersenne Twister 伪随机数引擎
static std::uniform_real_distribution<float> dis(0.0, 1.0);  // 区间 [0.0, 1.0)

template <typename Tdist, typename Tcorr>
static Tdist L2SqrRef(const Tcorr* a, const Tcorr* b, size_t d) {
  Tdist sum = 0;
  for (size_t i = 0; i < d; i++) {
    Tdist diff = Tdist(a[i]) - Tdist(b[i]);
    sum += diff * diff;
  }
  return sum;
}

HNSWLIB_avx512 static float L2Sqr_AVX512(const float* a, const float* b, size_t d) {
  __m512 msum512 = _mm512_setzero_ps();
  while (d >= 16) {
    __m512 ma = _mm512_loadu_ps(a);
    a += 16;
    __m512 mb = _mm512_loadu_ps(b);
    b += 16;
    __m512 mdiff = _mm512_sub_ps(ma, mb);
    msum512 = _mm512_add_ps(msum512, _mm512_mul_ps(mdiff, mdiff));
    d -= 16;
  }
  if (d == 0) {
    return _mm512_reduce_add_ps(msum512);
  }
  __m256 msum256 = _mm512_extractf32x8_ps(msum512, 1);
  msum256 = _mm256_add_ps(msum256, _mm512_extractf32x8_ps(msum512, 0));
  if (d >= 8) {
    __m256 ma = _mm256_loadu_ps(a);
    a += 8;
    __m256 mb = _mm256_loadu_ps(b);
    b += 8;
    __m256 mdiff = _mm256_sub_ps(ma, mb);
    msum256 = _mm256_add_ps(msum256, _mm256_mul_ps(mdiff, mdiff));
    d -= 8;
  }
  __m128 msum128 = _mm256_extractf128_ps(msum256, 1);
  msum128 = _mm_add_ps(msum128, _mm256_extractf128_ps(msum256, 0));
  if (d >= 4) {
    __m128 ma = _mm_loadu_ps(a);
    a += 4;
    __m128 mb = _mm_loadu_ps(b);
    b += 4;
    __m128 mdiff = _mm_sub_ps(ma, mb);
    msum128 = _mm_add_ps(msum128, _mm_mul_ps(mdiff, mdiff));
    d -= 4;
  }
  msum128 = _mm_hadd_ps(msum128, msum128);
  msum128 = _mm_hadd_ps(msum128, msum128);
  float sum = _mm_cvtss_f32(msum128);
  return d ? sum + L2SqrRef<float, float>(a, b, d) : sum;
}

static float* generate_embs(size_t data_num, size_t dim) {
  float* embs = (float*)malloc(data_num * dim * sizeof(float));

  for (size_t i = 0, k = 0; i < data_num; ++i) {
    for (size_t j = 0; j < dim; ++j, ++k) {
      embs[k] = dis(gen);
    }
  }

  return embs;
}

void test_avx512_l2_space() {
  size_t dim = 128;
  size_t data_num = 100;
  data_num *= 10000;

  float* query = generate_embs(1, dim);
  float* datas = generate_embs(data_num, dim);
  std::unordered_map<int, std::vector<float>> data_map;
  for (size_t i = 0; i < data_num; ++i) {
    data_map[i] = std::vector<float>();
    for (size_t j = 0; j < dim; ++j) {
      data_map[i].push_back(*datas);
      datas += 1;
    }
  }

  float* res = (float*)malloc(data_num * sizeof(float));

  auto s_search = std::chrono::steady_clock::now();
  for (size_t i = 0; i < data_num; ++i) {
    res[i] = L2Sqr_AVX512(query, data_map[i].data(), dim);
  }
  auto e_search = std::chrono::steady_clock::now();

  float sum = 0;
  for (size_t i = 0; i < data_num; ++i) {
    sum += res[i];
  }
  std::cout << "sum : " << sum << std::endl;
  std::cout << "avx512 l2 cost: " << time_cost(s_search, e_search) / 1000000 << " ms" << std::endl;
}