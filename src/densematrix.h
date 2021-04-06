/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>
#include <cstring>

#include <assert.h>
#include <immintrin.h>
#include "matrix.h"
#include "real.h"
#include "vector.h"

#define LINE_SIZE 64

namespace fasttext {

class Vector;

class DenseMatrix : public Matrix {
 protected:
  real* data_;
  uint64_t padded_n_;
  uint64_t nvec_;

 public:
  DenseMatrix();
  explicit DenseMatrix(int64_t, int64_t);
  //DenseMatrix(const DenseMatrix&) = default;
  //DenseMatrix(DenseMatrix&&) noexcept;
  
  DenseMatrix(const DenseMatrix& other) {
      if (&other != this) {
        if (data_)
          free(data_);
        padded_n_ = other.padded_n_;
        n_ = other.n_;
        m_ = other.m_;
        int res = posix_memalign((void**)&data_, LINE_SIZE, padded_n_ * m_ * sizeof(real));
        std::memcpy(data_, other.data_, padded_n_ * m_ * sizeof(real));
      }
  }
  
  DenseMatrix(DenseMatrix&& other) noexcept {
    if (&other != this) {
      if (data_)
        free(data_);
      data_ = other.data_;
      padded_n_ = other.padded_n_;
      n_ = other.n_;
      m_ = other.m_;
      other.padded_n_ = 0;
      other.n_ = 0;
      other.m_ = 0;
      other.data_ = nullptr;
    }
  }
  
  
  DenseMatrix& operator=(const DenseMatrix&) = delete;
  DenseMatrix& operator=(DenseMatrix&&) = delete;
  virtual ~DenseMatrix() noexcept override;

  inline real* data() {
    return data_;
  }
  inline const real* data() const {
    return data_;
  }

  inline const real& at(int64_t i, int64_t j) const {
    //assert(i * n_ + j < data_.size());
    return data_[i * padded_n_ + j];
  };
  inline real& at(int64_t i, int64_t j) {
    return data_[i * padded_n_ + j];
  };

  inline int64_t rows() const {
    return m_;
  }
  inline int64_t cols() const {
    return n_;
  }
  void zero();
  void uniform(real);

  void multiplyRow(const Vector& nums, int64_t ib = 0, int64_t ie = -1);
  void divideRow(const Vector& denoms, int64_t ib = 0, int64_t ie = -1);

  real l2NormRow(int64_t i) const;
  void l2NormRow(Vector& norms) const;

  real dotRow(const Vector&, int64_t) const override;
  void dotRows(const Vector& vec, Vector& target) const override;
  void addVectorToRow(const Vector&, int64_t, real) override;
  void addRowToVector(Vector& x, int32_t i) const override;
  void addRowToVector(Vector& x, int32_t i, real a) const override;
  void addVectorToRows(const Vector&, std::vector<int32_t>::const_iterator,
      std::vector<int32_t>::const_iterator, real) override;
  void addRowsToVector(Vector& x, std::vector<int32_t>::const_iterator start,
      std::vector<int32_t>::const_iterator end) const override;
  void addRowsToVector(Vector& x, std::vector<int32_t>::const_iterator start,
      std::vector<int32_t>::const_iterator end, real a) const override;
  void save(std::ostream&) const override;
  void load(std::istream&) override;
  void dump(std::ostream&) const override;
  
  void doBackpropSoftmax(Vector& grad, Vector& hidden, Vector& output, int32_t target, real lr) override;
  void doBackpropBinaryLogistic(Vector& grad, Vector& hidden, real score, real label, int32_t target, real lr) override;
};
} // namespace fasttext
