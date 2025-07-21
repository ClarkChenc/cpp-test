
#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <iostream>

// void sse_test() {
//   float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

//   for (size_t i = 0; i < sizeof(data) / sizeof(data[0]); i += 4) {
//     __m128 vec = _mm_loadu_ps(data + i);  // Load 4 floats into a SIMD register
//     float result[4];
//     _mm_storeu_ps(result, vec);  // Store the result back to an array
//     for (size_t j = 0; j < 4; ++j) {
//       std::cout << result[j] << " ";
//     }
//   }
// }

int horizontal_add_epi32(__m256i v) {
  // 将 __m256i 拆成两个 __m128i
  __m128i lo = _mm256_castsi256_si128(v);       // 低 128 位
  __m128i hi = _mm256_extracti128_si256(v, 1);  // 高 128 位

  // 对低高两个部分分别水平加
  lo = _mm_add_epi32(lo, hi);   // 合并为一个 __m128i（前四加后四）
  lo = _mm_hadd_epi32(lo, lo);  // [a0+a1, a2+a3, ...]
  lo = _mm_hadd_epi32(lo, lo);  // [a0+a1+a2+a3, ...]

  return _mm_cvtsi128_si32(lo);
}

template <typename Tdist, typename Tcorr>
static Tdist InnerProductRef(const Tcorr* a, const Tcorr* b, size_t d) {
  Tdist sum = 0;
  for (size_t i = 0; i < d; i++) {
    sum += Tdist(a[i]) * Tdist(b[i]);
  }
  return sum;
}

static float InnerProduct_SSE(const float* a, const float* b, size_t d) {
  __m128 msum128 = _mm_setzero_ps();
  while (d >= 4) {
    __m128 ma = _mm_loadu_ps(a);
    a += 4;
    __m128 mb = _mm_loadu_ps(b);
    b += 4;
    msum128 = _mm_add_ps(msum128, _mm_mul_ps(ma, mb));
    d -= 4;
  }
  msum128 = _mm_hadd_ps(msum128, msum128);
  msum128 = _mm_hadd_ps(msum128, msum128);
  float sum = _mm_cvtss_f32(msum128);
  return d ? sum + InnerProductRef<float, float>(a, b, d) : sum;
}

void sse_test() {
  // float data[] = {1.0f, 2.0f};

  // __m128 a = _mm_set1_ps(data[0]);    // Set all elements of a to data[0]
  // __m128 b = _mm_set1_ps(data[1]);    // Set all elements of b to data[1]
  // __m128 v1 = _mm_unpacklo_ps(a, b);  // Unpack low elements

  // float result[4];
  // _mm_storeu_ps(result, v1);
  // for (size_t j = 0; j < 4; ++j) {
  //   std::cout << result[j] << " ";
  // }

  // __m256i vec = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  // int ret = horizontal_add_epi32(vec);
  // std::cout << ret << std::endl;

  float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
  size_t data_dim = 4;

  float ret = -InnerProduct_SSE(a, b, data_dim);
  std::cout << ret << std ::endl;
}
