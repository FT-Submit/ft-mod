/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include <vector>

#include "matrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class Loss {
 private:
  void findKBest(
      int32_t k,
      real threshold,
      Predictions& heap,
      const Vector& output) const;

 protected:
  std::vector<real> t_sigmoid_;
  std::vector<real> t_log_;
  std::shared_ptr<Matrix>& wo_;
  uint32_t negctr;

  real log(real x) const;
  //real sigmoid(real x) const;
  
  std::vector<std::vector<int32_t>> paths_;
  std::vector<std::vector<bool>> codes_;

 public:
  explicit Loss(std::shared_ptr<Matrix>& wo);
  virtual ~Loss() = default;

  virtual real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) = 0;
  virtual real forward(
    const std::vector<int32_t>& positives,
    const std::vector<int32_t>& negatives,
    Model::State& state,
    real lr,
    bool backprop) { }
  virtual void computeOutput(Model::State& state) const = 0;

  virtual void predict(
      int32_t /*k*/,
      real /*threshold*/,
      Predictions& /*heap*/,
      Model::State& /*state*/) const;
      
  real sigmoid(real x) const;
  std::vector<real>& get_t_sigmoid() {
    return t_sigmoid_;
  }
  std::vector<std::vector<int32_t>>& get_paths() {
    return paths_;
  }
  std::vector<std::vector<bool>>& get_codes() {
    return codes_;
  }
  virtual int32_t getNegative(int32_t target, std::minstd_rand& rng) { };
  virtual int32_t sharedNegatives(std::vector<int32_t>& negatives, const std::vector<int32_t>& targets, std::minstd_rand& rng) { };
};

class BinaryLogisticLoss : public Loss {
 protected:
  real binaryLogistic(
      int32_t target,
      Model::State& state,
      bool labelIsPositive,
      real lr,
      bool backprop) const;

 public:
  explicit BinaryLogisticLoss(std::shared_ptr<Matrix>& wo);
  virtual ~BinaryLogisticLoss() noexcept override = default;
  void computeOutput(Model::State& state) const override;
};

class OneVsAllLoss : public BinaryLogisticLoss {
 public:
  explicit OneVsAllLoss(std::shared_ptr<Matrix>& wo);
  ~OneVsAllLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
};

class NegativeSamplingLoss : public BinaryLogisticLoss {
 protected:
  static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

  int neg_;
  std::vector<int32_t> negatives_;
  std::uniform_int_distribution<size_t> uniform_;
  int32_t getNegative(int32_t target, std::minstd_rand& rng) override;

 public:
  explicit NegativeSamplingLoss(
      std::shared_ptr<Matrix>& wo,
      int neg,
      const std::vector<int64_t>& targetCounts);
  ~NegativeSamplingLoss() noexcept override = default;

  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
    
  real forward(
    const std::vector<int32_t>& positives,
    const std::vector<int32_t>& negatives,
    Model::State& state,
    real lr,
    bool backprop) override;
    
  int32_t sharedNegatives(std::vector<int32_t>& negatives, const std::vector<int32_t>& targets, std::minstd_rand& rng) override;
};

class HierarchicalSoftmaxLoss : public BinaryLogisticLoss {
 protected:
  struct Node {
    int32_t parent;
    int32_t left;
    int32_t right;
    int64_t count;
    bool binary;
  };

  //std::vector<std::vector<int32_t>> paths_;
  //std::vector<std::vector<bool>> codes_;
  std::vector<Node> tree_;
  int32_t osz_;
  void buildTree(const std::vector<int64_t>& counts);
  void dfs(
      int32_t k,
      real threshold,
      int32_t node,
      real score,
      Predictions& heap,
      const Vector& hidden) const;

 public:
  explicit HierarchicalSoftmaxLoss(
      std::shared_ptr<Matrix>& wo,
      const std::vector<int64_t>& counts);
  ~HierarchicalSoftmaxLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
  void predict(
      int32_t k,
      real threshold,
      Predictions& heap,
      Model::State& state) const override;
};

class SoftmaxLoss : public Loss {
 public:
  explicit SoftmaxLoss(std::shared_ptr<Matrix>& wo);
  ~SoftmaxLoss() noexcept override = default;
  real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) override;
  void computeOutput(Model::State& state) const override;
};

} // namespace fasttext