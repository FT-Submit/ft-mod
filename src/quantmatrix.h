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

#include <memory>
#include <vector>

#include "real.h"

#include "densematrix.h"
#include "matrix.h"
#include "vector.h"

#include "productquantizer.h"
#include <immintrin.h>

namespace fasttext {

class QuantMatrix : public Matrix {
 protected:
  std::unique_ptr<ProductQuantizer> pq_;
  std::unique_ptr<ProductQuantizer> npq_;

  std::vector<uint8_t> codes_;
  std::vector<uint8_t> norm_codes_;

  bool qnorm_;
  int32_t codesize_;

 public:
  QuantMatrix();
  QuantMatrix(DenseMatrix&&, int32_t, bool);
  QuantMatrix(const QuantMatrix&) = delete;
  QuantMatrix(QuantMatrix&&) = delete;
  QuantMatrix& operator=(const QuantMatrix&) = delete;
  QuantMatrix& operator=(QuantMatrix&&) = delete;
  virtual ~QuantMatrix() noexcept override = default;

  void quantizeNorm(const Vector&);
  void quantize(DenseMatrix&& mat);

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
  real placeholder;
  inline real& at(int64_t i, int64_t j) { return placeholder; }
};

} // namespace fasttext
