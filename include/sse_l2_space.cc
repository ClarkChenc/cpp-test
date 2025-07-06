
#include <iostream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <vector>

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "utils.h"

std::random_device rd;                                // 用于生成种子（可替换为固定种子）
std::mt19937 gen(rd());                               // Mersenne Twister 伪随机数引擎
std::uniform_real_distribution<float> dis(0.0, 1.0);  // 区间 [0.0, 1.0)

static inline float horizontal_add(__m128 v) {
  __m128 sum1 = _mm_hadd_ps(v, v);        // [a+b, c+d, a+b, c+d]
  __m128 sum2 = _mm_hadd_ps(sum1, sum1);  // [a+b+c+d, a+b+c+d, ...]
  return _mm_cvtss_f32(sum2);             // 取第一个元素
}

float L2SqrSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr

) {
  float* pVect1 = (float*)pVect1v;
  float* pVect2 = (float*)pVect2v;
  size_t qty = *((size_t*)qty_ptr);

  size_t qty16 = qty >> 4;
  qty16 = qty16 << 4;

  const float* pEnd1 = pVect1 + qty16;

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    // 每一遍计算 16 个 float

    _mm_prefetch((char*)(pVect1 + 64), _MM_HINT_T0);
    _mm_prefetch((char*)(pVect2 + 64), _MM_HINT_T0);

    for (int i = 0; i < 4; ++i) {
      v1 = _mm_loadu_ps(pVect1);
      pVect1 += 4;
      v2 = _mm_loadu_ps(pVect2);
      pVect2 += 4;
      diff = _mm_sub_ps(v1, v2);
      sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
  }

  // _mm_store_ps(TmpRes, sum);
  return horizontal_add(sum);
}

float* generate_embs(size_t data_num, size_t dim) {
  float* embs = (float*)malloc(data_num * dim * sizeof(float));

  for (size_t i = 0, k = 0; i < data_num; ++i) {
    for (size_t j = 0; j < dim; ++j, ++k) {
      embs[k] = dis(gen);
    }
  }

  return embs;
}

void test_l2_space() {
  size_t dim = 128;
  size_t data_num = 100;
  data_num *= 10000;

  float* query = generate_embs(1, dim);
  float* datas = generate_embs(data_num, dim);
  std::unordered_map<int, std::vector<float>> data_map;
  for (size_t i = 0; i < data_num; ++i) {
    data_map[i] = std::vector<float>(datas + i * dim, datas + (i + 1) * dim);
  }

  float* res = (float*)malloc(data_num * sizeof(float));

  auto s_search = std::chrono::steady_clock::now();
  for (size_t i = 0; i < data_num; ++i) {
    res[i] = L2SqrSIMD16ExtSSE(query, &data_map[i], &dim);
  }
  auto e_search = std::chrono::steady_clock::now();

  float sum = 0;
  for (size_t i = 0; i < data_num; ++i) {
    sum += res[i];
  }
  std::cout << "sum : " << sum << std::endl;
  std::cout << "cost: " << time_cost(s_search, e_search) / 1000000 << " ms" << std::endl;
}