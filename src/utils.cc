/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"

#include <ios>

namespace fasttext {

namespace utils {

int64_t size(std::ifstream& ifs) {
  ifs.seekg(std::streamoff(0), std::ios::end);
  return ifs.tellg();
}

void seek(std::ifstream& ifs, int64_t pos) {
  ifs.clear();
  ifs.seekg(std::streampos(pos));
}
} // namespace utils

} // namespace fasttext

float _mm512_reduce_add_ps(__m512 v) {
  float out[16];
  __m512 v1 = _mm512_permute_ps(v, 0xb1);
  __m512 s1 = _mm512_add_ps(v, v1);
  __m512 v2 = _mm512_permute_ps(s1, 0x4e);
  __m512 s2 = _mm512_add_ps(s1, v2);
  __m512 v3 = _mm512_shuffle_f32x4(s2, s2, 0xb1);
  __m512 s3 = _mm512_add_ps(s2, v3);
  __m512 v4 = _mm512_shuffle_f32x4(s3, s3, 0x4e);
  __m512 s4 = _mm512_add_ps(s3, v4);
  _mm512_storeu_ps(out, s4);
  return out[0];
}

__m512 _mm512_abs_ps(__m512 v) {
  __m512 mone = _mm512_set1_ps(-1);
  __m512 neg = _mm512_mul_ps(v, mone);
  return _mm512_max_ps(v, neg);
}
