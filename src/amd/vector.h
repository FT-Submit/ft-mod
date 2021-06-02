/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <ostream>
#include <vector>
#include <cstring>
#include <iostream>

#include "real.h"

#include <immintrin.h>

#define AVX true
#define BLOCKED true
#define LARGE_BLOCKS true
#define REALS_PER_LINE 16
#define LINE_SIZE 64


namespace fasttext {

class Matrix;

class Vector {
 protected:
  real* data_;
  uint64_t l_;
  uint64_t lvec_;

 public:
  explicit Vector(int64_t);
  //Vector(const Vector&) = default;
  //Vector(Vector&&) noexcept = default;
  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) = default;
  
  Vector(const Vector& other) {
      if (&other != this) {
        if (data_)
          free(data_);
        l_ = other.l_;
        lvec_ = other.lvec_;
        int res = posix_memalign((void**)&data_, LINE_SIZE, lvec_ * sizeof(real));
        std::memcpy(data_, other.data_, lvec_ * sizeof(real));
      }
  }
  
  Vector(Vector&& other) {
    if (&other != this) {
      if (data_)
        free(data_);
      data_ = other.data_;
      l_ = other.l_;
      lvec_ = other.lvec_;
      other.l_ = 0;
      other.lvec_ = 0;
      other.data_ = nullptr;
    }
  }
  //Vector& operator=(const Vector&);
  //Vector& operator=(Vector&&);
  ~Vector();

  inline real* data() {
    return data_;
  }
  inline const real* data() const {
    return data_;
  }
  inline real& operator[](int64_t i) {
    return data_[i];
  }
  inline const real& operator[](int64_t i) const {
    return data_[i];
  }

  inline int64_t size() const {
    return l_;
  }
  void clear();
  void zero();
  void mul(real);
  real norm() const;
  void addVector(const Vector& source);
  void addVector(const Vector&, real);
  void addRow(const Matrix&, int64_t);
  void addRow(const Matrix&, int64_t, real);
  void addRows(const Matrix&, std::vector<int32_t>::const_iterator,
      std::vector<int32_t>::const_iterator);
  void addRows(const Matrix&, std::vector<int32_t>::const_iterator,
      std::vector<int32_t>::const_iterator, real);
  void mul(const Matrix&, const Vector&);
  int64_t argmax();
  
};

std::ostream& operator<<(std::ostream&, const Vector&);

#if AVX
float _mm256_reduce_add_ps(__m256 v);
__m256 _mm256_abs_ps(__m256 v);
#endif
	
} // namespace fasttext



