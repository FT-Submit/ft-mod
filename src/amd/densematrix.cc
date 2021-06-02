/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "densematrix.h"

#include <exception>
#include <random>
#include <stdexcept>
#include <utility>

#include <iostream>

#include "utils.h"
#include "vector.h"

namespace fasttext {

DenseMatrix::DenseMatrix() : DenseMatrix(0, 0) {
  data_ = nullptr;
  padded_n_ = 0;
}

DenseMatrix::DenseMatrix(int64_t m, int64_t n) : Matrix(m, n) {
  padded_n_ = (n_ + (REALS_PER_LINE - 1)) / REALS_PER_LINE * REALS_PER_LINE;
  int res = posix_memalign((void**)&data_, LINE_SIZE, padded_n_ * m_ * sizeof(real));
}

DenseMatrix::~DenseMatrix() {
  if (data_)
    free(data_);
}

void DenseMatrix::zero() {
  std::fill(data_, data_ + padded_n_ * m_, 0.0);
}

void DenseMatrix::uniform(real a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (uint64_t i = 0; i < m_; ++i)
    for (uint64_t j = 0; j < n_; ++j)
      data_[i * padded_n_ + j] = uniform(rng);
}

void DenseMatrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  //assert(ie <= nums.size());
  for (auto i = ib; i < ie; i++) {
    real n = nums[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) *= n;
      }
    }
  }
}

void DenseMatrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  //assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    real n = denoms[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

real DenseMatrix::l2NormRow(int64_t i) const {
  auto norm = 0.0;
  for (auto j = 0; j < n_; j++) {
    norm += at(i, j) * at(i, j);
  }
  if (std::isnan(norm)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return std::sqrt(norm);
}

void DenseMatrix::l2NormRow(Vector& norms) const {
  //assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

real DenseMatrix::dotRow(const Vector& vec, int64_t i) const {
  real d = 0.0;
  real d1 = 0.0;
  real d2 = 0.0;
  real d3 = 0.0;
  real d4 = 0.0;
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();
  __m256 acc7 = _mm256_setzero_ps();
  __m256 acc8 = _mm256_setzero_ps();
  int64_t n8 = n_ - (n_ % 64);
  int64_t n6 = n_ - ((n_ - n8) % 48);
  for (int64_t j = 0; j < n8; j += 64) {
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j]), _mm256_loadu_ps(&vec[j]), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 8]), _mm256_loadu_ps(&vec[j + 8]), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 16]), _mm256_loadu_ps(&vec[j + 16]), acc3);
    acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 24]), _mm256_loadu_ps(&vec[j + 24]), acc4);
    acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 32]), _mm256_loadu_ps(&vec[j + 32]), acc5);
    acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 40]), _mm256_loadu_ps(&vec[j + 40]), acc6);
    acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 48]), _mm256_loadu_ps(&vec[j + 48]), acc7);
    acc8 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 56]), _mm256_loadu_ps(&vec[j + 56]), acc8);
  }
  for (int64_t j = n8; j < n6; j += 48) {
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j]), _mm256_loadu_ps(&vec[j]), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 8]), _mm256_loadu_ps(&vec[j + 8]), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 16]), _mm256_loadu_ps(&vec[j + 16]), acc3);
    acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 24]), _mm256_loadu_ps(&vec[j + 24]), acc4);
    acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 32]), _mm256_loadu_ps(&vec[j + 32]), acc5);
    acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 40]), _mm256_loadu_ps(&vec[j + 40]), acc6);
  }
  for (int64_t j = n6; j < padded_n_; j += 8) {
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j]), _mm256_loadu_ps(&vec[j]), acc1);
  }
  acc1 = _mm256_add_ps(acc1, acc2);
  acc3 = _mm256_add_ps(acc3, acc4);
  acc5 = _mm256_add_ps(acc5, acc6);
  acc7 = _mm256_add_ps(acc7, acc8);
  acc1 = _mm256_add_ps(acc1, acc3);
  acc5 = _mm256_add_ps(acc5, acc7);
  acc1 = _mm256_add_ps(acc1, acc5);
  return _mm256_reduce_add_ps(acc1);
  
  if (std::isnan(d)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return d + d1 + d2 + d3 + d4;
}
 
//can also try 32
#define BLOCK_SIZE 32

void DenseMatrix::dotRows(const Vector& vec, Vector& target) const {
  
  uint64_t block_n = n_ / BLOCK_SIZE;
  uint64_t block_m = m_ / BLOCK_SIZE;
  
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  
  target.zero();
  for (uint64_t ib = 0; ib < m_; ib += BLOCK_SIZE) {
    for (uint64_t jb = 0; jb < n_; jb += BLOCK_SIZE) {
      for (uint64_t ib1 = 0; ib1 < BLOCK_SIZE; ++ib1) {
        uint64_t offset = (ib + ib1) * padded_n_ + jb;
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[offset]), _mm256_loadu_ps(&vec[jb]), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[offset + 8]), _mm256_loadu_ps(&vec[jb + 8]), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[offset + 16]), _mm256_loadu_ps(&vec[jb + 16]), acc3);
        acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[offset + 24]), _mm256_loadu_ps(&vec[jb + 24]), acc4);
        target[ib + ib1] += _mm256_reduce_add_ps(_mm256_add_ps(_mm256_add_ps(acc1, acc2), _mm256_add_ps(acc3, acc4)));        
      }
    }
    for (uint64_t ib1 = 0; ib1 < BLOCK_SIZE; ++ib1) {
      for (uint64_t j = n_ - block_n * BLOCK_SIZE; j < n_; ++j) {
        target[ib + ib1] += at(ib + ib1, j) * vec[j];
      }
    }
  }
  for (uint64_t jb = 0; jb < n_; jb += BLOCK_SIZE) {
    for (uint64_t i = m_ - block_m * BLOCK_SIZE; i < m_; ++i) {
      uint64_t offset = i * padded_n_ + jb;
      acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[offset]), _mm256_loadu_ps(&vec[jb]), acc1);
      acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[offset + 8]), _mm256_loadu_ps(&vec[jb + 8]), acc2);
      acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[offset + 16]), _mm256_loadu_ps(&vec[jb + 16]), acc3);
      acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(&data_[offset + 24]), _mm256_loadu_ps(&vec[jb + 24]), acc4);
      target[i] += _mm256_reduce_add_ps(_mm256_add_ps(_mm256_add_ps(acc1, acc2), _mm256_add_ps(acc3, acc4)));        
    }
  }
  for (uint64_t i = m_ - block_m * BLOCK_SIZE; i < m_; ++i) {
    for (uint64_t j = n_ - block_n * BLOCK_SIZE; j < n_; ++j) {
      target[i] += at(i, j) * vec[j];
    }
  }
  
}

void DenseMatrix::addVectorToRow(const Vector& vec, int64_t i, real a) {
  __m256 avec = _mm256_set1_ps(a);
  for (int64_t j = 0; j < padded_n_; j += 8) {
    _mm256_storeu_ps(&data_[i * padded_n_ + j], _mm256_fmadd_ps(
        _mm256_loadu_ps(&vec[j]), avec, _mm256_loadu_ps(&data_[i * padded_n_ + j])));
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i) const {
  for (int64_t j = 0; j < padded_n_; j += 8) {
    _mm256_storeu_ps(&x[j], _mm256_add_ps(
        _mm256_loadu_ps(&x[j]), _mm256_loadu_ps(&data_[i * padded_n_ + j])));
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i, real a) const {
  __m256 avec = _mm256_set1_ps(a);
  for (int64_t j = 0; j < padded_n_; j += 8) {
    _mm256_storeu_ps(&x[j], _mm256_fmadd_ps(
        avec, _mm256_loadu_ps(&data_[i * padded_n_ + j]), _mm256_loadu_ps(&x[j])));
  }
}

void DenseMatrix::addVectorToRows(const Vector& vec, std::vector<int32_t>::const_iterator start,
    std::vector<int32_t>::const_iterator end, real a) {
  __m256 avec = _mm256_set1_ps(a);
  __m256 vvec;
  __m256 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m256 vvec7, vvec8, vvec9, vvec10, vvec11, vvec12;
  __m256 vvec13, vvec14, vvec15, vvec16, vvec17, vvec18, vvec19;
  int64_t n192 = padded_n_ - (padded_n_ % 96);
  int64_t n32 = padded_n_ - ((padded_n_ - n192) % 16);
  for (int64_t j = 0; j < n192; j += 96) {
    vvec1 = _mm256_loadu_ps(&vec[j]);
    vvec2 = _mm256_loadu_ps(&vec[j + 8]);
    vvec3 = _mm256_loadu_ps(&vec[j + 16]);
    vvec4 = _mm256_loadu_ps(&vec[j + 24]);
    vvec5 = _mm256_loadu_ps(&vec[j + 32]);
    vvec6 = _mm256_loadu_ps(&vec[j + 40]);
    vvec7 = _mm256_loadu_ps(&vec[j + 48]);
    vvec8 = _mm256_loadu_ps(&vec[j + 56]);
    vvec9 = _mm256_loadu_ps(&vec[j + 64]);
    vvec10 = _mm256_loadu_ps(&vec[j + 72]);
    vvec11 = _mm256_loadu_ps(&vec[j + 80]);
    vvec12 = _mm256_loadu_ps(&vec[j + 88]);
    for (auto ipt = start; ipt != end; ++ipt) {
      uint32_t i32 = *ipt;
      uint64_t i = i32;
      _mm256_storeu_ps(&data_[i * padded_n_ + j], _mm256_fmadd_ps(
          vvec1, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 8], _mm256_fmadd_ps(
          vvec2, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 8])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 16], _mm256_fmadd_ps(
          vvec3, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 16])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 24], _mm256_fmadd_ps(
          vvec4, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 24])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 32], _mm256_fmadd_ps(
          vvec5, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 32])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 40], _mm256_fmadd_ps(
          vvec6, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 40])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 48], _mm256_fmadd_ps(
          vvec7, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 48])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 56], _mm256_fmadd_ps(
          vvec8, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 56])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 64], _mm256_fmadd_ps(
          vvec9, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 64])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 72], _mm256_fmadd_ps(
          vvec10, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 72])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 80], _mm256_fmadd_ps(
          vvec11, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 80])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 88], _mm256_fmadd_ps(
          vvec12, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 88])));
    }
  }
  for (int64_t j = n192; j < n32; j += 16) {
    vvec1 = _mm256_loadu_ps(&vec[j]);
    vvec2 = _mm256_loadu_ps(&vec[j + 8]);
    for (auto ipt = start; ipt != end; ++ipt) {
      uint32_t i32 = *ipt;
      uint64_t i = i32;
      _mm256_storeu_ps(&data_[i * padded_n_ + j], _mm256_fmadd_ps(
          vvec1, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j])));
      _mm256_storeu_ps(&data_[i * padded_n_ + j + 8], _mm256_fmadd_ps(
          vvec2, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 8])));
    }
  }
  for (int64_t j = n32; j < padded_n_; j += 8) {
    vvec = _mm256_loadu_ps(&vec[j]);
    for (auto ipt = start; ipt != end; ++ipt) {
      uint32_t i32 = *ipt;
      uint64_t i = i32;
      _mm256_storeu_ps(&data_[i * padded_n_ + j], _mm256_fmadd_ps(
          vvec, avec, _mm256_loadu_ps(&data_[i * padded_n_ + j])));
    }
  }
}

void DenseMatrix::addRowsToVector(Vector& x, std::vector<int32_t>::const_iterator start,
    std::vector<int32_t>::const_iterator end) const {
  __m256 vvec;
  __m256 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m256 vvec7, vvec8, vvec9, vvec10, vvec11, vvec12;
  __m256 vvec13, vvec14, vvec15, vvec16, vvec17, vvec18, vvec19;
  int64_t n128 = padded_n_ - (padded_n_ % 64);
  int64_t n96 = padded_n_ - ((padded_n_ - n128) % 48);
  for (int64_t j = 0; j < n128; j += 64) {
    vvec1 = _mm256_loadu_ps(&x[j]);
    vvec2 = _mm256_loadu_ps(&x[j + 8]);
    vvec3 = _mm256_loadu_ps(&x[j + 16]);
    vvec4 = _mm256_loadu_ps(&x[j + 24]);
    vvec5 = _mm256_loadu_ps(&x[j + 32]);
    vvec6 = _mm256_loadu_ps(&x[j + 40]);
    vvec7 = _mm256_loadu_ps(&x[j + 48]);
    vvec8 = _mm256_loadu_ps(&x[j + 56]);
    for (auto ipt = start; ipt != end; ++ipt) {
      int32_t i32 = *ipt;
      int64_t i = i32;      
      vvec1 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j]), vvec1);
      vvec2 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 8]), vvec2);
      vvec3 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 16]), vvec3);
      vvec4 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 24]), vvec4);
      vvec5 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 32]), vvec5);
      vvec6 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 40]), vvec6);
      vvec7 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 48]), vvec7);
      vvec8 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 56]), vvec8);
    }
    _mm256_storeu_ps(&x[j], vvec1);
    _mm256_storeu_ps(&x[j + 8], vvec2);
    _mm256_storeu_ps(&x[j + 16], vvec3);
    _mm256_storeu_ps(&x[j + 24], vvec4);
    _mm256_storeu_ps(&x[j + 32], vvec5);
    _mm256_storeu_ps(&x[j + 40], vvec6);
    _mm256_storeu_ps(&x[j + 48], vvec7);
    _mm256_storeu_ps(&x[j + 56], vvec8);
  }
  for (int64_t j = n128; j < n96; j += 48) {
    vvec1 = _mm256_loadu_ps(&x[j]);
    vvec2 = _mm256_loadu_ps(&x[j + 8]);
    vvec3 = _mm256_loadu_ps(&x[j + 16]);
    vvec4 = _mm256_loadu_ps(&x[j + 24]);
    vvec5 = _mm256_loadu_ps(&x[j + 32]);
    vvec6 = _mm256_loadu_ps(&x[j + 40]);
    for (auto ipt = start; ipt != end; ++ipt) {
      int32_t i32 = *ipt;
      int64_t i = i32;      
      vvec1 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j]), vvec1);
      vvec2 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 8]), vvec2);
      vvec3 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 16]), vvec3);
      vvec4 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 24]), vvec4);
      vvec5 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 32]), vvec5);
      vvec6 = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j + 40]), vvec6);
    }
    _mm256_storeu_ps(&x[j], vvec1);
    _mm256_storeu_ps(&x[j + 8], vvec2);
    _mm256_storeu_ps(&x[j + 16], vvec3);
    _mm256_storeu_ps(&x[j + 24], vvec4);
    _mm256_storeu_ps(&x[j + 32], vvec5);
    _mm256_storeu_ps(&x[j + 40], vvec6);
  }
  for (int64_t j = n96; j < padded_n_; j += 8) {
    vvec = _mm256_loadu_ps(&x[j]);
    for (auto ipt = start; ipt != end; ++ipt) {
      int32_t i32 = *ipt;
      int64_t i = i32;
      vvec = _mm256_add_ps(_mm256_loadu_ps(&data_[i * padded_n_ + j]), vvec);
    }
    _mm256_storeu_ps(&x[j], vvec);
  }
}

void DenseMatrix::addRowsToVector(Vector& x, std::vector<int32_t>::const_iterator start,
    std::vector<int32_t>::const_iterator end, real a) const {
  __m256 vvec;
  __m256 avec = _mm256_set1_ps(a);
  __m256 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m256 vvec7, vvec8, vvec9, vvec10, vvec11, vvec12;
  __m256 vvec13, vvec14, vvec15, vvec16, vvec17, vvec18, vvec19;
  int64_t n128 = padded_n_ - (padded_n_ % 64);
  int64_t n96 = padded_n_ - ((padded_n_ - n128) % 48);
  for (int64_t j = 0; j < n128; j += 64) {
    vvec1 = _mm256_loadu_ps(&x[j]);
    vvec2 = _mm256_loadu_ps(&x[j + 8]);
    vvec3 = _mm256_loadu_ps(&x[j + 16]);
    vvec4 = _mm256_loadu_ps(&x[j + 24]);
    vvec5 = _mm256_loadu_ps(&x[j + 32]);
    vvec6 = _mm256_loadu_ps(&x[j + 40]);
    vvec7 = _mm256_loadu_ps(&x[j + 48]);
    vvec8 = _mm256_loadu_ps(&x[j + 56]);
    for (auto ipt = start; ipt != end; ++ipt) {
      int32_t i32 = *ipt;
      int64_t i = i32;      
      vvec1 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j]), vvec1);
      vvec2 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 8]), vvec2);
      vvec3 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 16]), vvec3);
      vvec4 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 24]), vvec4);
      vvec5 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 32]), vvec5);
      vvec6 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 40]), vvec6);
      vvec7 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 48]), vvec7);
      vvec8 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 56]), vvec8);
    }
    _mm256_storeu_ps(&x[j], vvec1);
    _mm256_storeu_ps(&x[j + 8], vvec2);
    _mm256_storeu_ps(&x[j + 16], vvec3);
    _mm256_storeu_ps(&x[j + 24], vvec4);
    _mm256_storeu_ps(&x[j + 32], vvec5);
    _mm256_storeu_ps(&x[j + 40], vvec6);
    _mm256_storeu_ps(&x[j + 48], vvec7);
    _mm256_storeu_ps(&x[j + 56], vvec8);
  }
  for (int64_t j = n128; j < n96; j += 48) {
    vvec1 = _mm256_loadu_ps(&x[j]);
    vvec2 = _mm256_loadu_ps(&x[j + 8]);
    vvec3 = _mm256_loadu_ps(&x[j + 16]);
    vvec4 = _mm256_loadu_ps(&x[j + 24]);
    vvec5 = _mm256_loadu_ps(&x[j + 32]);
    vvec6 = _mm256_loadu_ps(&x[j + 40]);
    for (auto ipt = start; ipt != end; ++ipt) {
      int32_t i32 = *ipt;
      int64_t i = i32;      
      vvec1 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j]), vvec1);
      vvec2 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 8]), vvec2);
      vvec3 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 16]), vvec3);
      vvec4 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 24]), vvec4);
      vvec5 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 32]), vvec5);
      vvec6 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j + 40]), vvec6);
    }
    _mm256_storeu_ps(&x[j], vvec1);
    _mm256_storeu_ps(&x[j + 8], vvec2);
    _mm256_storeu_ps(&x[j + 16], vvec3);
    _mm256_storeu_ps(&x[j + 24], vvec4);
    _mm256_storeu_ps(&x[j + 32], vvec5);
    _mm256_storeu_ps(&x[j + 40], vvec6);
  }
  for (int64_t j = n96; j < padded_n_; j += 8) {
    vvec = _mm256_loadu_ps(&x[j]);
    for (auto ipt = start; ipt != end; ++ipt) {
      int32_t i32 = *ipt;
      int64_t i = i32;
      vvec = _mm256_fmadd_ps(avec, _mm256_loadu_ps(&data_[i * padded_n_ + j]), vvec);
    }
    _mm256_storeu_ps(&x[j], vvec);
  }
}

void DenseMatrix::doBackpropSoftmax(Vector& grad, Vector& hidden, Vector& output, int32_t target, real lr) {
  __m256 gvec, hvec, avec, dvec;
  int32_t osz = size(0);
  int64_t n16 = this->size(1) - (this->size(1) & 7);
  for (int64_t j = 0; j < n16; j += 8) {
    gvec = _mm256_loadu_ps(&grad[j]);
    hvec = _mm256_loadu_ps(&hidden[j]);
    for (uint32_t ipt = 0; ipt != osz; ++ipt) {
      uint64_t i = ipt;
      real label = (ipt == target) ? 1.0 : 0.0;
      real alpha = lr * (label - output[ipt]);
      avec = _mm256_set1_ps(alpha);
      dvec = _mm256_loadu_ps(&data_[i * padded_n_ + j]);
      gvec = _mm256_fmadd_ps(avec, dvec, gvec);
      _mm256_storeu_ps(&data_[i * padded_n_ + j], _mm256_fmadd_ps(hvec, avec, dvec));
    }
    _mm256_storeu_ps(&grad[j], gvec);
  }
  for (uint32_t ipt = 0; ipt != osz; ++ipt) {
    uint64_t i = ipt;
    real label = (ipt == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output[ipt]);
    for (int64_t j = n16; j < this->size(1); j++) {
      grad[j] += alpha * data_[i * padded_n_ + j];
      data_[i * padded_n_ + j] += alpha * hidden[j];
    }
  }
}

void DenseMatrix::doBackpropBinaryLogistic(Vector& grad, Vector& hidden, real score, real label, int32_t target, real lr) {
  __m256 gvec, hvec, avec, dvec, rvec;
  real alpha = lr * (label - score);
  avec = _mm256_set1_ps(alpha);
  for (int64_t j = 0; j < padded_n_; j += 8) {
    dvec = _mm256_loadu_ps(&data_[target * padded_n_ + j]);
    gvec = _mm256_loadu_ps(&grad[j]);
    hvec = _mm256_loadu_ps(&hidden[j]);
    rvec = _mm256_fmadd_ps(avec, dvec, gvec);
    _mm256_storeu_ps(&data_[target * padded_n_ + j], _mm256_fmadd_ps(hvec, avec, dvec));
    _mm256_storeu_ps(&grad[j], rvec);
  }
}

void DenseMatrix::save(std::ostream& out) const {
  out.write((char*)&m_, sizeof(int64_t));
  out.write((char*)&n_, sizeof(int64_t));
  for (uint64_t i = 0; i < m_; ++i)
    out.write((char*)(data_ + i * padded_n_), n_ * sizeof(real));
}

void DenseMatrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  padded_n_ = (n_ + (REALS_PER_LINE - 1)) / REALS_PER_LINE * REALS_PER_LINE;
  if(data_)
    free(data_);
  int res = posix_memalign((void**)&data_, LINE_SIZE, padded_n_ * m_ * sizeof(real));
  for (uint64_t i = 0; i < m_; ++i)
    in.read((char*)(data_ + i * padded_n_), n_ * sizeof(real));
  
}

void DenseMatrix::dump(std::ostream& out) const {
  out << m_ << " " << n_ << std::endl;
  for (int64_t i = 0; i < m_; i++) {
    for (int64_t j = 0; j < n_; j++) {
      if (j > 0) {
        out << " ";
      }
      out << at(i, j);
    }
    out << std::endl;
  }
};

} // namespace fasttext
