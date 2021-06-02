/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "vector.h"

#include <assert.h>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <utility>

#include "matrix.h"

#if STORE_IN_AVX

namespace fasttext {

Vector::Vector(int64_t m) : l_(m), lvec_((m + 15) / 16) {
  int res = posix_memalign((void**)&data_, REALS_PER_LINE, lvec_ * sizeof(__m512));
}

Vector::~Vector() {
  if (data_)
    free(data_);
  data_ = nullptr;
}

void Vector::zero() {
  for (int64_t i = 0; i < lvec_; ++i)
    data_[i] = _mm512_setzero_ps();
}

real Vector::norm() const {
  __m512 sum = _mm512_setzero_ps();
  for (int64_t i = 0; i < lvec_; i++) {
    sum = _mm512_fmadd_ps(data_[i], data_[i], sum);
  }
  return std::sqrt(_mm512_reduce_add_ps(sum));
}

void Vector::mul(real a) {
  __m512 avec = _mm512_set1_ps(a);
  for (int64_t i = 0; i < lvec_; i++) {
    data_[i] = _mm512_mul_ps(data_[i], avec);
  }
}

void Vector::addVector(const Vector& source) {
  for (int64_t i = 0; i < lvec_; i++) {
    data_[i] = _mm512_add_ps(data_[i], source.data_[i]);
  }
}

void Vector::addVector(const Vector& source, real s) {
  __m512 svec = _mm512_set1_ps(s);
  for (int64_t i = 0; i < lvec_; i++) {
    data_[i] = _mm512_fmadd_ps(source.data_[i], svec, data_[i]);
  }
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
  A.addRowToVector(*this, i, a);
}

void Vector::addRow(const Matrix& A, int64_t i) {
  A.addRowToVector(*this, i);
}

//this is new
void Vector::addRows(const Matrix& A, std::vector<int32_t>::const_iterator start,
    std::vector<int32_t>::const_iterator end, real a) {
  A.addRowsToVector(*this, start, end, a);
}

void Vector::addRows(const Matrix& A, std::vector<int32_t>::const_iterator start,
    std::vector<int32_t>::const_iterator end) {
  A.addRowsToVector(*this, start, end);
}

void Vector::mul(const Matrix& A, const Vector& vec) {
  A.dotRows(vec, *this);
}

int64_t Vector::argmax() {
  // this is very inefficient, but it's only for inference
  real max = _mm512_reduce_max_ps(data_[0]);
  int64_t large_idx = 0;
  for (int64_t i = 1; i < lvec_; i++) {
    real candidate = _mm512_reduce_max_ps(data_[i]);
    if (candidate > max) {
      max = candidate;
      large_idx = i;
    }
  }
  real arr[16];
  _mm512_storeu_ps(arr, data_[large_idx]);
  for (int64_t i = 0; i < 16; i++) {
    //comparison of floats which should be identical - no room for rounding errors
    if (arr[i] == max) {
      return large_idx * 16 + i;
    }
  }
  //just to be sure
  real max_safe = arr[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < 16; i++) {
    if (arr[i] > max_safe) {
      max_safe = arr[i];
      argmax = i;
    }
  }
  return large_idx * 16 + argmax;
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << std::setprecision(5);
  real arr[16];
  for (int64_t j = 0; j < v.lvec(); ++j) {
    _mm512_storeu_ps(arr, v[j]);
    for (int64_t jj = 0; jj < 16; ++jj) {
      if (j * 16 + jj < v.size()) {
        os << arr[jj] << ' ';
      }
    }
  }
  return os;
}

} // namespace fasttext

#else

namespace fasttext {

//Vector::Vector(int64_t m) : data_(m) {}

Vector::Vector(int64_t m) : l_(m), lvec_((m + 7) / 8 * 8) {
  int res = posix_memalign((void**)&data_, REALS_PER_LINE, lvec_ * sizeof(real));
}

Vector::~Vector() {
  if (data_)
    free(data_);
  data_ = nullptr;
}

void Vector::zero() {
  //std::fill(data_.begin(), data_.end(), 0.0);
  std::fill(data_, data_ + lvec_, 0.0);
}

real Vector::norm() const {
  real sum = 0;
  for (int64_t i = 0; i < size(); i++) {
    sum += data_[i] * data_[i];
  }
  return std::sqrt(sum);
}

void Vector::mul(real a) {
  #if AVX
  int64_t vsize = size();
  //int64_t vsize16 = vsize - (vsize & 15);
  __m256 avec = _mm256_set1_ps(a);
  for (int64_t i = 0; i < lvec_; i += 8) {
    _mm256_storeu_ps(&data_[i], _mm256_mul_ps(_mm256_loadu_ps(&data_[i]), avec));
  }
  //for (int64_t i = vsize16; i < vsize; i++) {
  //  data_[i] *= a;
  //}
  #else
  for (int64_t i = 0; i < size(); i++) {
    data_[i] *= a;
  }
  #endif
}

void Vector::addVector(const Vector& source) {
  //assert(size() == source.size());
  #if AVX
  //int64_t vsize = size();
  //int64_t vsize16 = vsize - (vsize & 15);
  for (int64_t i = 0; i < lvec_; i += 16) {
    _mm256_storeu_ps(&data_[i], _mm256_add_ps(_mm256_loadu_ps(&data_[i]),
        _mm256_loadu_ps(&source.data_[i])));
  }
  //for (int64_t i = vsize16; i < vsize; i++) {
  //  data_[i] += source.data_[i];
  //}
  #else
  for (int64_t i = 0; i < size(); i++) {
    data_[i] += source.data_[i];
  }
  #endif
}

void Vector::addVector(const Vector& source, real s) {
  //assert(size() == source.size());
  #if AVX
  //int64_t vsize = size();
  //int64_t vsize16 = vsize - (vsize & 15);
  __m256 svec = _mm256_set1_ps(s);
  for (int64_t i = 0; i < lvec_; i += 8) {
    _mm256_storeu_ps(&data_[i], _mm256_fmadd_ps(_mm256_loadu_ps(&source.data_[i]),
        svec, _mm256_loadu_ps(&data_[i])));
  }
  //for (int64_t i = vsize16; i < vsize; i++) {
  //  data_[i] += s * source.data_[i];
  //}
  #else
  for (int64_t i = 0; i < size(); i++) {
    data_[i] += s * source.data_[i];
  }
  #endif
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
  //assert(i >= 0);
  //assert(i < A.size(0));
  //assert(size() == A.size(1));
  A.addRowToVector(*this, i, a);
}

void Vector::addRow(const Matrix& A, int64_t i) {
  //assert(i >= 0);
  //assert(i < A.size(0));
  //assert(size() == A.size(1));
  A.addRowToVector(*this, i);
}

//this is new
void Vector::addRows(const Matrix& A, std::vector<int32_t>::const_iterator start,
    std::vector<int32_t>::const_iterator end, real a) {
  //assert(start >= 0);
  //assert(end < A.size(0));
  //assert(size() == A.size(1));
  A.addRowsToVector(*this, start, end, a);
}

void Vector::addRows(const Matrix& A, std::vector<int32_t>::const_iterator start,
    std::vector<int32_t>::const_iterator end) {
  //assert(start >= 0);
  //assert(end < A.size(0));
  //assert(size() == A.size(1));
  A.addRowsToVector(*this, start, end);
}
//end of new

void Vector::mul(const Matrix& A, const Vector& vec) {
  //assert(A.size(0) == size());
  //assert(A.size(1) == vec.size());
  //for (int64_t i = 0; i < size(); i++) {
  //  data_[i] = A.dotRow(vec, i);
  //}
  A.dotRows(vec, *this);
}

int64_t Vector::argmax() {
  real max = data_[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < size(); i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.size(); j++) {
    os << v[j] << ' ';
  }
  return os;
}

#if AVX
float _mm256_reduce_add_ps(__m256 v) {
  float out[8];
  __m256 v1 = _mm256_permute_ps(v, 0xb1);
  __m256 s1 = _mm256_add_ps(v, v1);
  __m256 v2 = _mm256_permute_ps(s1, 0x4e);
  __m256 s2 = _mm256_add_ps(s1, v2);
  __m256 v3 = _mm256_permute2f128_ps(s2, s2 , 0x01);
  __m256 s3 = _mm256_add_ps(s2, v3);
  _mm256_storeu_ps(out, s3);
  return out[0];
}

__m256 _mm256_abs_ps(__m256 v) {
  __m256 mone = _mm256_set1_ps(-1);
  __m256 neg = _mm256_mul_ps(v, mone);
  return _mm256_max_ps(v, neg);
}
#endif


} // namespace fasttext


#endif
