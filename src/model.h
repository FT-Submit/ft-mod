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
#include <utility>
#include <vector>

#include "matrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class Loss;

class Model {
 protected:
  std::shared_ptr<Matrix> wi_;
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<Loss> loss_;
  bool normalizeGradient_;

 public:
  Model(
      std::shared_ptr<Matrix> wi,
      std::shared_ptr<Matrix> wo,
      std::shared_ptr<Loss> loss,
      bool normalizeGradient);
  Model(const Model& model) = delete;
  Model(Model&& model) = delete;
  Model& operator=(const Model& other) = delete;
  Model& operator=(Model&& other) = delete;

  class State {
   private:
    real lossValue_;
    int64_t nexamples_;

   public:
    Vector hidden;
    Vector output;
    Vector grad;
    std::minstd_rand rng;

    State(int32_t hiddenSize, int32_t outputSize, int32_t seed);
    real getLoss() const;
    void incrementNExamples(real loss);
    void incrementNExamples(real loss, int32_t inc);
  };

  void predict(
      const std::vector<int32_t>& input,
      int32_t k,
      real threshold,
      Predictions& heap,
      State& state) const;
  void update(
      const std::vector<int32_t>& input,
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      real lr,
      State& state);
  void update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& positives,
    const std::vector<int32_t>& negatives,
    real lr,
    State& state);
  void updateFirst(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state);
  void updateMid(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state);
  void updateLast(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state);
  void computeHidden(const std::vector<int32_t>& input, State& state) const;

  static const int32_t kUnlimitedPredictions = -1;
  static const int32_t kAllLabelsAsTarget = -1;
  
  const std::shared_ptr<Loss> getLoss() {
    return loss_;
  }
  
  void updateNoHidden(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state);
    
  void updateNoHidden(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& positives,
    const std::vector<int32_t>& negatives,
    real lr,
    State& state);
  
  void removeFromHidden(const std::vector<int32_t>& input, State& state);
  void addToHidden(const std::vector<int32_t>& input, State& state);
  void addTo(const std::vector<int32_t>& input, Vector& gradient, State& state);
  void denormalizeHidden(int32_t factor, State& state);
  void normalizeHidden(int32_t factor, State& state);
  void fixHiddenGradient(int32_t factor, State& state);
  void addGradient(const std::vector<int32_t>& input, State& state);
};

} // namespace fasttext
