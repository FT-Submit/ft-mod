/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fasttext.h"
#include "loss.h"
#include "quantmatrix.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fasttext {

constexpr int32_t FASTTEXT_VERSION = 12; /* Version 1b */
constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

bool comparePairs(
    const std::pair<real, std::string>& l,
    const std::pair<real, std::string>& r);

std::shared_ptr<Loss> FastText::createLoss(std::shared_ptr<Matrix>& output) {
  loss_name lossName = args_->loss;
  switch (lossName) {
    case loss_name::hs:
      return std::make_shared<HierarchicalSoftmaxLoss>(
          output, getTargetCounts());
    case loss_name::ns:
      return std::make_shared<NegativeSamplingLoss>(
          output, args_->neg, getTargetCounts());
    case loss_name::softmax:
      return std::make_shared<SoftmaxLoss>(output);
    case loss_name::ova:
      return std::make_shared<OneVsAllLoss>(output);
    default:
      throw std::runtime_error("Unknown loss");
  }
}

FastText::FastText() : quant_(false), wordVectors_(nullptr) { }

void FastText::addInputVector(Vector& vec, int32_t ind) const {
  vec.addRow(*input_, ind);
}

std::shared_ptr<const Dictionary> FastText::getDictionary() const {
  return dict_;
}

const Args FastText::getArgs() const {
  return *args_.get();
}

std::shared_ptr<const DenseMatrix> FastText::getInputMatrix() const {
  if (quant_) {
    throw std::runtime_error("Can't export quantized matrix");
  }
  assert(input_.get());
  return std::dynamic_pointer_cast<DenseMatrix>(input_);
}

std::shared_ptr<const DenseMatrix> FastText::getOutputMatrix() const {
  if (quant_ && args_->qout) {
    throw std::runtime_error("Can't export quantized matrix");
  }
  assert(output_.get());
  return std::dynamic_pointer_cast<DenseMatrix>(output_);
}

int32_t FastText::getWordId(const std::string& word) const {
  return dict_->getId(word);
}

int32_t FastText::getSubwordId(const std::string& subword) const {
  int32_t h = dict_->hash(subword) % args_->bucket;
  return dict_->nwords() + h;
}

void FastText::getWordVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
  vec.zero();
  for (int i = 0; i < ngrams.size(); i++) {
    addInputVector(vec, ngrams[i]);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void FastText::getVector(Vector& vec, const std::string& word) const {
  getWordVector(vec, word);
}

void FastText::getSubwordVector(Vector& vec, const std::string& subword) const {
  vec.zero();
  int32_t h = dict_->hash(subword) % args_->bucket;
  h = h + dict_->nwords();
  addInputVector(vec, h);
}

void FastText::saveVectors(const std::string& filename) {
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        filename + " cannot be opened for saving vectors!");
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveVectors() {
  saveVectors(args_->output + ".vec");
}

void FastText::saveOutput(const std::string& filename) {
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        filename + " cannot be opened for saving vectors!");
  }
  if (quant_) {
    throw std::invalid_argument(
        "Option -saveOutput is not supported for quantized models.");
  }
  int32_t n =
      (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
  ofs << n << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < n; i++) {
    std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
                                                         : dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveOutput() {
  saveOutput(args_->output + ".output");
}

bool FastText::checkModel(std::istream& in) {
  int32_t magic;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version > FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

void FastText::signModel(std::ostream& out) {
  const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
  const int32_t version = FASTTEXT_VERSION;
  out.write((char*)&(magic), sizeof(int32_t));
  out.write((char*)&(version), sizeof(int32_t));
}

void FastText::saveModel() {
  std::string fn(args_->output);
  if (quant_) {
    fn += ".ftz";
  } else {
    fn += ".bin";
  }
  saveModel(fn);
}

void FastText::saveModel(const std::string& filename) {
  std::ofstream ofs(filename, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for saving!");
  }
  signModel(ofs);
  args_->save(ofs);
  dict_->save(ofs);

  ofs.write((char*)&(quant_), sizeof(bool));
  input_->save(ofs);

  ofs.write((char*)&(args_->qout), sizeof(bool));
  output_->save(ofs);

  ofs.close();
}

void FastText::loadModel(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
}

std::vector<int64_t> FastText::getTargetCounts() const {
  if (args_->model == model_name::sup) {
    return dict_->getCounts(entry_type::label);
  } else {
    return dict_->getCounts(entry_type::word);
  }
}

void FastText::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  input_ = std::make_shared<DenseMatrix>();
  output_ = std::make_shared<DenseMatrix>();
  args_->load(in);
  if (version == 11 && args_->model == model_name::sup) {
    // backward compatibility: old supervised models do not use char ngrams.
    args_->maxn = 0;
  }
  dict_ = std::make_shared<Dictionary>(args_, in);

  bool quant_input;
  in.read((char*)&quant_input, sizeof(bool));
  if (quant_input) {
    quant_ = true;
    input_ = std::make_shared<QuantMatrix>();
  }
  input_->load(in);

  if (!quant_input && dict_->isPruned()) {
    throw std::invalid_argument(
        "Invalid model file.\n"
        "Please download the updated model from www.fasttext.cc.\n"
        "See issue #332 on Github for more information.\n");
  }

  in.read((char*)&args_->qout, sizeof(bool));
  if (quant_ && args_->qout) {
    output_ = std::make_shared<QuantMatrix>();
  }
  output_->load(in);

  auto loss = createLoss(output_);
  bool normalizeGradient = (args_->model == model_name::sup);
  model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
}

void FastText::printInfo(real progress, real loss, std::ostream& log_stream) {
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double t =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start_)
          .count();
  double lr = args_->lr * (1.0 - progress);
  double wst = 0;

  int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)

  if (progress > 0 && t >= 0) {
    progress = progress * 100;
    eta = t * (100 - progress) / progress;
    wst = double(tokenCount_) / t / args_->thread;
  }
  int32_t etah = eta / 3600;
  int32_t etam = (eta % 3600) / 60;

  log_stream << std::fixed;
  log_stream << "Progress: ";
  log_stream << std::setprecision(1) << std::setw(5) << progress << "%";
  log_stream << " words/sec/thread: " << std::setw(7) << int64_t(wst);
  log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
  log_stream << " loss: " << std::setw(9) << std::setprecision(6) << loss;
  log_stream << " ETA: " << std::setw(3) << etah;
  log_stream << "h" << std::setw(2) << etam << "m";
  log_stream << std::flush;
}

std::vector<int32_t> FastText::selectEmbeddings(int32_t cutoff) const {
  #if STORE_IN_AVX
    std::cout << "Not supported\n";
    std::vector<int32_t> idx(0);
    return idx;
  #else
  std::shared_ptr<DenseMatrix> input =
      std::dynamic_pointer_cast<DenseMatrix>(input_);
  Vector norms(input->size(0));
  input->l2NormRow(norms);
  std::vector<int32_t> idx(input->size(0), 0);
  std::iota(idx.begin(), idx.end(), 0);
  auto eosid = dict_->getId(Dictionary::EOS);
  std::sort(idx.begin(), idx.end(), [&norms, eosid](size_t i1, size_t i2) {
    return eosid == i1 || (eosid != i2 && norms[i1] > norms[i2]);
  });
  idx.erase(idx.begin() + cutoff, idx.end());
  return idx;
  #endif
}

void FastText::quantize(const Args& qargs) {
  if (args_->model != model_name::sup) {
    throw std::invalid_argument(
        "For now we only support quantization of supervised models");
  }
  args_->input = qargs.input;
  args_->qout = qargs.qout;
  args_->output = qargs.output;
  std::shared_ptr<DenseMatrix> input =
      std::dynamic_pointer_cast<DenseMatrix>(input_);
  std::shared_ptr<DenseMatrix> output =
      std::dynamic_pointer_cast<DenseMatrix>(output_);
  bool normalizeGradient = (args_->model == model_name::sup);

  if (qargs.cutoff > 0 && qargs.cutoff < input->size(0)) {
    auto idx = selectEmbeddings(qargs.cutoff);
    dict_->prune(idx);
    std::shared_ptr<DenseMatrix> ninput =
        std::make_shared<DenseMatrix>(idx.size(), args_->dim);
    for (auto i = 0; i < idx.size(); i++) {
      for (auto j = 0; j < args_->dim; j++) {
        ninput->at(i, j) = input->at(idx[i], j);
      }
    }
    input = ninput;
    if (qargs.retrain) {
      args_->epoch = qargs.epoch;
      args_->lr = qargs.lr;
      args_->thread = qargs.thread;
      args_->verbose = qargs.verbose;
      auto loss = createLoss(output_);
      model_ = std::make_shared<Model>(input, output, loss, normalizeGradient);
      startThreads();
    }
  }

  input_ = std::make_shared<QuantMatrix>(
      std::move(*(input.get())), qargs.dsub, qargs.qnorm);

  if (args_->qout) {
    output_ = std::make_shared<QuantMatrix>(
        std::move(*(output.get())), 2, qargs.qnorm);
  }

  quant_ = true;
  auto loss = createLoss(output_);
  model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
}

void FastText::supervised(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line,
    const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) {
    return;
  }
  if (args_->loss == loss_name::ova) {
    model_->update(line, labels, Model::kAllLabelsAsTarget, lr, state);
  } else {
    std::uniform_int_distribution<> uniform(0, labels.size() - 1);
    int32_t i = uniform(state.rng);
    model_->update(line, labels, i, lr, state);
  }
}

void FastText::cbow(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model_->update(bow, line, w, lr, state);
  }
}

void FastText::cbow_shared(
    int32_t threadId,
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  int32_t sharedLen = args_->shared == 0 ? args_->ws * 2 + 1 : args_->shared;
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  std::vector<int32_t> negatives(args_->neg);
  std::vector<int32_t> positives(sharedLen);
  std::vector<int32_t> one_positive(1);
  auto loss = model_->getLoss();
  int32_t get_neg_ctr = sharedLen;
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    if (get_neg_ctr == sharedLen) {
      int32_t i = 0;
      while (i < sharedLen && w + i < line.size()) {
        positives[i] = line[w + i];
        ++i;
      }
      while (i < sharedLen) {
        positives[i] = -1;
        ++i;
      }
      loss->sharedNegatives(negatives, positives, state.rng);
      get_neg_ctr = 0;
    }
    one_positive[0] = line[w];
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model_->update(bow, one_positive, negatives, lr, state);
    ++get_neg_ctr;
  }
  
}

// gradMap holds delayed gradient idxs
// gradMat holds delayed gradients
// On remove: check if gradMat has entry for idx. If yes, add grad to M_i and zero entry
// On update: check if gradMat has entry for idx. If yes, update delayed grad, else create entry and assign grad
// On add:    no delayed gradient yet - no action
void FastText::cbow_uphid(
    int32_t threadId,
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line,
    std::vector<int32_t>& gradMap,
    std::vector<Vector>& gradMat) {
  std::vector<int32_t> bow;
  std::vector<int32_t> old_bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  int32_t old_boundary_l = -1;
  int32_t old_boundary_r = -1; 
  int32_t old_nglen, current_nglen, remaining_nglen;
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    // normalization: keep old, de-normalize by old, get new, normalize by new
    current_nglen = bow.size();
    remaining_nglen = 0;
    if (old_boundary_r >= 0) {
      // denormalize hidden layer
      model_->denormalizeHidden(old_nglen, state);
      // remove all left to the new boundary
      for (int32_t i = std::max(0, old_boundary_l); i < w - boundary && i <= old_boundary_r; ++i) {
        if (i != w - 1) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          model_->removeFromHidden(ngrams, state);
          model_->addGradient(ngrams, state);
          int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
          if (idx < gradMap.size()) {
            model_->addTo(ngrams, gradMat[idx], state);
            gradMap[idx] = -1;
          }
        }
      }
      // remove all right to the new boundary
      for (int32_t i = w + boundary + 1; i < line.size() && i <= old_boundary_r; ++i) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
        model_->removeFromHidden(ngrams, state);
        model_->addGradient(ngrams, state);
        int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
        if (idx < gradMap.size()) {
          model_->addTo(ngrams, gradMat[idx], state);
          gradMap[idx] = -1;
        }
      }
      // remove w, if needed
      if (w <= old_boundary_r) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
        model_->removeFromHidden(ngrams, state);
        model_->addGradient(ngrams, state);
        int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
        if (idx < gradMap.size()) {
          model_->addTo(ngrams, gradMat[idx], state);
          gradMap[idx] = -1;
        }
      }
      // keep all within the intersection of boundaries
      for (int32_t i = std::max(0, std::max(old_boundary_l, w - boundary)); i < line.size() && i <= old_boundary_r && i <= w + boundary; ++i) {
        if (i != w - 1 && i != w) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          remaining_nglen += ngrams.size();
          int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
          if (idx < gradMap.size()) {
            gradMat[idx].addVector(state.grad, 1.0);
          } else {
            int32_t i = 0;
            while (i < gradMap.size() && gradMap[i] >= 0) ++i;
            gradMap[i] = ngrams[0];
            real* dat = gradMat[i].data();
            int32_t dim = args_->dim;
            for (int32_t j = 0; j < dim; ++j) dat[j] = state.grad[j];
          }
        }
      }
      model_->fixHiddenGradient(remaining_nglen, state);
      // add w - 1, if needed
      if (w - 1 >= w - boundary) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w - 1]);
        model_->addToHidden(ngrams, state);
      }
      // add all left to the old boundary
      for (int32_t i = std::max(w - boundary, 0); i < old_boundary_l; ++i) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
        model_->addToHidden(ngrams, state);
      }
      // add all right to the old boundary
      for (int32_t i = old_boundary_r + 1; i < line.size() && i <= w + boundary; ++i) {
        if (i != w) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          model_->addToHidden(ngrams, state);
        }
      }
      // re-normalize hidden
      model_->normalizeHidden(current_nglen, state);
      // update
      model_->updateNoHidden(bow, line, w, lr, state);
    } else {
      model_->addToHidden(bow, state);
      model_->normalizeHidden(current_nglen, state);
      model_->updateNoHidden(bow, line, w, lr, state);
    }
    old_boundary_l = w - boundary;
    old_boundary_r = w + boundary;
    old_nglen = current_nglen;
    
    old_bow = bow;
  }
  
  if (old_boundary_r >= 0) {
    model_->addGradient(bow, state);
  }
  for (int32_t i = 0; i < 2 * args_->ws + 1; ++i) {
    if (gradMap[i] >= 0) model_->addTo(bow, gradMat[i], state);
    gradMap[i] = -1;
  }
  
}

void FastText::cbow_uphid_fixed(
    int32_t threadId,
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line,
    std::vector<int32_t>& gradMap,
    std::vector<Vector>& gradMat) {
  int32_t sharedLen = 2 * args_->ws + 1;
  std::vector<int32_t> bow;
  std::vector<int32_t> old_bow;
  int32_t old_boundary_l = -1;
  int32_t old_boundary_r = -1; 
  int32_t old_nglen, current_nglen, remaining_nglen;
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = args_->ws;
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    // normalization: keep old, de-normalize by old, get new, normalize by new
    current_nglen = bow.size();
    remaining_nglen = 0;
    if (old_boundary_r >= 0) {
      // denormalize hidden layer
      model_->denormalizeHidden(old_nglen, state);
      // remove all left to the new boundary
      for (int32_t i = std::max(0, old_boundary_l); i < w - boundary && i <= old_boundary_r; ++i) {
        if (i != w - 1) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          model_->removeFromHidden(ngrams, state);
          model_->addGradient(ngrams, state);
          int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
          if (idx < gradMap.size()) {
            model_->addTo(ngrams, gradMat[idx], state);
            gradMap[idx] = -1;
          }
        }
      }
      // remove all right to the new boundary
      for (int32_t i = w + boundary + 1; i < line.size() && i <= old_boundary_r; ++i) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
        model_->removeFromHidden(ngrams, state);
        model_->addGradient(ngrams, state);
        int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
        if (idx < gradMap.size()) {
          model_->addTo(ngrams, gradMat[idx], state);
          gradMap[idx] = -1;
        }
      }
      // remove w, if needed
      if (w <= old_boundary_r) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
        model_->removeFromHidden(ngrams, state);
        model_->addGradient(ngrams, state);
        int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
        if (idx < gradMap.size()) {
          model_->addTo(ngrams, gradMat[idx], state);
          gradMap[idx] = -1;
        }
      }
      // keep all within the intersection of boundaries
      for (int32_t i = std::max(0, std::max(old_boundary_l, w - boundary)); i < line.size() && i <= old_boundary_r && i <= w + boundary; ++i) {
        if (i != w - 1 && i != w) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          remaining_nglen += ngrams.size();
          int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
          if (idx < gradMap.size()) {
            gradMat[idx].addVector(state.grad, 1.0);
          } else {
            int32_t i = 0;
            while (i < gradMap.size() && gradMap[i] >= 0) ++i;
            gradMap[i] = ngrams[0];
            real* dat = gradMat[i].data();
            int32_t dim = args_->dim;
            for (int32_t j = 0; j < dim; ++j) dat[j] = state.grad[j];
          }
        }
      }
      model_->fixHiddenGradient(remaining_nglen, state);
      // add w - 1, if needed
      if (w - 1 >= w - boundary) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w - 1]);
        model_->addToHidden(ngrams, state);
      }
      // add all left to the old boundary
      for (int32_t i = std::max(w - boundary, 0); i < old_boundary_l; ++i) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
        model_->addToHidden(ngrams, state);
      }
      // add all right to the old boundary
      for (int32_t i = old_boundary_r + 1; i < line.size() && i <= w + boundary; ++i) {
        if (i != w) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          model_->addToHidden(ngrams, state);
        }
      }
      // re-normalize hidden
      model_->normalizeHidden(current_nglen, state);
      // update
      model_->updateNoHidden(bow, line, w, lr, state);
    } else {
      model_->addToHidden(bow, state);
      model_->normalizeHidden(current_nglen, state);
      model_->updateNoHidden(bow, line, w, lr, state);
    }
    old_boundary_l = w - boundary;
    old_boundary_r = w + boundary;
    old_nglen = current_nglen;
    
    old_bow = bow;
  }
  
  if (old_boundary_r >= 0) {
    model_->addGradient(bow, state);
  }
  for (int32_t i = 0; i < 2 * args_->ws + 1; ++i) {
    if (gradMap[i] >= 0) model_->addTo(bow, gradMat[i], state);
    gradMap[i] = -1;
  }
  
}

void FastText::cbow_uphid_shared(
    int32_t threadId,
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line,
    std::vector<int32_t>& gradMap,
    std::vector<Vector>& gradMat) {
  int32_t sharedLen = args_->shared == 0 ? 2 * args_->ws + 1 : args_->shared;
  std::vector<int32_t> bow;
  std::vector<int32_t> old_bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  std::vector<int32_t> negatives(args_->neg);
  std::vector<int32_t> positives(sharedLen);
  std::vector<int32_t> one_positive(1);
  auto loss = model_->getLoss();
  int32_t get_neg_ctr = sharedLen;
  int32_t old_boundary_l = -1;
  int32_t old_boundary_r = -1; 
  int32_t old_nglen, current_nglen, remaining_nglen;
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    if (get_neg_ctr == sharedLen) {
      int32_t i = 0;
      while (i < sharedLen && w + i < line.size()) {
        positives[i] = line[w + i];
        ++i;
      }
      while (i < sharedLen) {
        positives[i] = -1;
        ++i;
      }
      loss->sharedNegatives(negatives, positives, state.rng);
      get_neg_ctr = 0;
    }
    one_positive[0] = line[w];
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    // normalization: keep old, de-normalize by old, get new, normalize by new
    current_nglen = bow.size();
    remaining_nglen = 0;
    if (old_boundary_r >= 0) {
      // denormalize hidden layer
      model_->denormalizeHidden(old_nglen, state);
      // remove all left to the new boundary
      for (int32_t i = std::max(0, old_boundary_l); i < w - boundary && i <= old_boundary_r; ++i) {
        if (i != w - 1) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          model_->removeFromHidden(ngrams, state);
          model_->addGradient(ngrams, state);
          int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
          if (idx < gradMap.size()) {
            model_->addTo(ngrams, gradMat[idx], state);
            gradMap[idx] = -1;
          }
        }
      }
      // remove all right to the new boundary
      for (int32_t i = w + boundary + 1; i < line.size() && i <= old_boundary_r; ++i) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
        model_->removeFromHidden(ngrams, state);
        model_->addGradient(ngrams, state);
        int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
        if (idx < gradMap.size()) {
          model_->addTo(ngrams, gradMat[idx], state);
          gradMap[idx] = -1;
        }
      }
      // remove w, if needed
      if (w <= old_boundary_r) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
        model_->removeFromHidden(ngrams, state);
        model_->addGradient(ngrams, state);
        int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
        if (idx < gradMap.size()) {
          model_->addTo(ngrams, gradMat[idx], state);
          gradMap[idx] = -1;
        }
      }
      // keep all within the intersection of boundaries
      for (int32_t i = std::max(0, std::max(old_boundary_l, w - boundary)); i < line.size() && i <= old_boundary_r && i <= w + boundary; ++i) {
        if (i != w - 1 && i != w) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          remaining_nglen += ngrams.size();
          int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
          if (idx < gradMap.size()) {
            gradMat[idx].addVector(state.grad, 1.0);
          } else {
            int32_t i = 0;
            while (i < gradMap.size() && gradMap[i] >= 0) ++i;
            gradMap[i] = ngrams[0];
            real* dat = gradMat[i].data();
            int32_t dim = args_->dim;
            for (int32_t j = 0; j < dim; ++j) dat[j] = state.grad[j];
          }
        }
      }
      model_->fixHiddenGradient(remaining_nglen, state);
      // add w - 1, if needed
      if (w - 1 >= w - boundary) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w - 1]);
        model_->addToHidden(ngrams, state);
      }
      // add all left to the old boundary
      for (int32_t i = std::max(w - boundary, 0); i < old_boundary_l; ++i) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
        model_->addToHidden(ngrams, state);
      }
      // add all right to the old boundary
      for (int32_t i = old_boundary_r + 1; i < line.size() && i <= w + boundary; ++i) {
        if (i != w) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          model_->addToHidden(ngrams, state);
        }
      }
      // re-normalize hidden
      model_->normalizeHidden(current_nglen, state);
      // update
      model_->updateNoHidden(bow, one_positive, negatives, lr, state);
    } else {
      model_->addToHidden(bow, state);
      model_->normalizeHidden(current_nglen, state);
      model_->updateNoHidden(bow, one_positive, negatives, lr, state);
    }
    old_boundary_l = w - boundary;
    old_boundary_r = w + boundary;
    old_nglen = current_nglen;
    
    old_bow = bow;
    
    ++get_neg_ctr;
  }
  
  if (old_boundary_r >= 0) {
    model_->addGradient(bow, state);
  }
  for (int32_t i = 0; i < 2 * args_->ws + 1; ++i) {
    if (gradMap[i] >= 0) model_->addTo(bow, gradMat[i], state);
    gradMap[i] = -1;
  }
  
}

void FastText::cbow_uphid_shared_fixed(
    int32_t threadId,
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line,
    std::vector<int32_t>& gradMap,
    std::vector<Vector>& gradMat) {
  int32_t sharedLen = args_->shared == 0 ? 2 * args_->ws + 1 : args_->shared;
  std::vector<int32_t> bow;
  std::vector<int32_t> old_bow;
  std::vector<int32_t> negatives(args_->neg);
  std::vector<int32_t> positives(sharedLen);
  std::vector<int32_t> one_positive(1);
  auto loss = model_->getLoss();
  int32_t get_neg_ctr = sharedLen;
  int32_t old_boundary_l = -1;
  int32_t old_boundary_r = -1; 
  int32_t old_nglen, current_nglen, remaining_nglen;
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = args_->ws;
    if (get_neg_ctr == sharedLen) {
      int32_t i = 0;
      while (i < sharedLen && w + i < line.size()) {
        positives[i] = line[w + i];
        ++i;
      }
      while (i < sharedLen) {
        positives[i] = -1;
        ++i;
      }
      loss->sharedNegatives(negatives, positives, state.rng);
      get_neg_ctr = 0;
    }
    one_positive[0] = line[w];
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    // normalization: keep old, de-normalize by old, get new, normalize by new
    current_nglen = bow.size();
    remaining_nglen = 0;
    if (old_boundary_r >= 0) {
      // denormalize hidden layer
      model_->denormalizeHidden(old_nglen, state);
      // remove all left to the new boundary
      for (int32_t i = std::max(0, old_boundary_l); i < w - boundary && i <= old_boundary_r; ++i) {
        if (i != w - 1) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          model_->removeFromHidden(ngrams, state);
          model_->addGradient(ngrams, state);
          int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
          if (idx < gradMap.size()) {
            model_->addTo(ngrams, gradMat[idx], state);
            gradMap[idx] = -1;
          }
        }
      }
      // remove all right to the new boundary
      for (int32_t i = w + boundary + 1; i < line.size() && i <= old_boundary_r; ++i) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
        model_->removeFromHidden(ngrams, state);
        model_->addGradient(ngrams, state);
        int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
        if (idx < gradMap.size()) {
          model_->addTo(ngrams, gradMat[idx], state);
          gradMap[idx] = -1;
        }
      }
      // remove w, if needed
      if (w <= old_boundary_r) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
        model_->removeFromHidden(ngrams, state);
        model_->addGradient(ngrams, state);
        int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
        if (idx < gradMap.size()) {
          model_->addTo(ngrams, gradMat[idx], state);
          gradMap[idx] = -1;
        }
      }
      // keep all within the intersection of boundaries
      for (int32_t i = std::max(0, std::max(old_boundary_l, w - boundary)); i < line.size() && i <= old_boundary_r && i <= w + boundary; ++i) {
        if (i != w - 1 && i != w) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          remaining_nglen += ngrams.size();
          int32_t idx = std::distance(gradMap.begin(), std::find(gradMap.begin(), gradMap.end(), ngrams[0]));
          if (idx < gradMap.size()) {
            gradMat[idx].addVector(state.grad, 1.0);
          } else {
            int32_t i = 0;
            while (i < gradMap.size() && gradMap[i] >= 0) ++i;
            gradMap[i] = ngrams[0];
            real* dat = gradMat[i].data();
            int32_t dim = args_->dim;
            for (int32_t j = 0; j < dim; ++j) dat[j] = state.grad[j];
          }
        }
      }
      model_->fixHiddenGradient(remaining_nglen, state);
      // add w - 1, if needed
      if (w - 1 >= w - boundary) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w - 1]);
        model_->addToHidden(ngrams, state);
      }
      // add all left to the old boundary
      for (int32_t i = std::max(w - boundary, 0); i < old_boundary_l; ++i) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
        model_->addToHidden(ngrams, state);
      }
      // add all right to the old boundary
      for (int32_t i = old_boundary_r + 1; i < line.size() && i <= w + boundary; ++i) {
        if (i != w) {
          const std::vector<int32_t>& ngrams = dict_->getSubwords(line[i]);
          model_->addToHidden(ngrams, state);
        }
      }
      // re-normalize hidden
      model_->normalizeHidden(current_nglen, state);
      // update
      model_->updateNoHidden(bow, one_positive, negatives, lr, state);
    } else {
      model_->addToHidden(bow, state);
      model_->normalizeHidden(current_nglen, state);
      model_->updateNoHidden(bow, one_positive, negatives, lr, state);
    }
    old_boundary_l = w - boundary;
    old_boundary_r = w + boundary;
    old_nglen = current_nglen;
    
    old_bow = bow;
    
    ++get_neg_ctr;
  }
  
  if (old_boundary_r >= 0) {
    model_->addGradient(bow, state);
  }
  for (int32_t i = 0; i < 2 * args_->ws + 1; ++i) {
    if (gradMap[i] >= 0) model_->addTo(bow, gradMat[i], state);
    gradMap[i] = -1;
  }
  
}

void FastText::skipgram(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model_->update(ngrams, line, w + c, lr, state);
      }
    }
  }
}

void FastText::skipgram_batched(
    int32_t threadId,
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    
    int32_t startc = std::max(-boundary, -w);
    if (startc == 0) startc++;
    int32_t endc = std::min(boundary, (int32_t)(line.size() - w - 1));
    if (endc == 0) endc--;
    if (startc == endc) {
      model_->update(ngrams, line, w + startc, lr, state);
    } else if (startc < endc) {
      model_->updateFirst(ngrams, line, w + startc, lr, state);
      for (int32_t c = startc + 1; c < endc; c++) {
        if (c != 0) {
          model_->updateMid(ngrams, line, w + c, lr, state);
        }
      }
      model_->updateLast(ngrams, line, w + startc, lr, state);
    }
  }
}

void FastText::skipgram_ns(
    int32_t threadId,
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  int32_t sharedLen = 2 * args_->ws + 1;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  std::vector<int32_t> negatives(args_->neg);
  std::vector<int32_t> positives(sharedLen);
  auto loss = model_->getLoss();
  int32_t get_neg_ctr = 0;
  loss->sharedNegatives(negatives, line, state.rng);
  for (int32_t w = 0; w < line.size(); w++) {
    get_neg_ctr = 0;
    int32_t boundary = uniform(state.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    int32_t j = 0;
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) positives[j] = line[w + c];
      else positives[j] = -1;
      ++j;
    }
    while (j < sharedLen) {
      positives[j] = -1;
      ++j;
    }
    loss->sharedNegatives(negatives, positives, state.rng);
    model_->update(ngrams, positives, negatives, lr, state);
  }
}

std::tuple<int64_t, double, double>
FastText::test(std::istream& in, int32_t k, real threshold) {
  Meter meter;
  test(in, k, threshold, meter);

  return std::tuple<int64_t, double, double>(
      meter.nexamples(), meter.precision(), meter.recall());
}

void FastText::test(std::istream& in, int32_t k, real threshold, Meter& meter)
    const {
  std::vector<int32_t> line;
  std::vector<int32_t> labels;
  Predictions predictions;

  while (in.peek() != EOF) {
    line.clear();
    labels.clear();
    dict_->getLine(in, line, labels);

    if (!labels.empty() && !line.empty()) {
      predictions.clear();
      predict(k, line, predictions, threshold);
      meter.log(labels, predictions);
    }
  }
}

void FastText::predict(
    int32_t k,
    const std::vector<int32_t>& words,
    Predictions& predictions,
    real threshold) const {
  if (words.empty()) {
    return;
  }
  Model::State state(args_->dim, dict_->nlabels(), 0);
  if (args_->model != model_name::sup) {
    throw std::invalid_argument("Model needs to be supervised for prediction!");
  }
  model_->predict(words, k, threshold, predictions, state);
}

bool FastText::predictLine(
    std::istream& in,
    std::vector<std::pair<real, std::string>>& predictions,
    int32_t k,
    real threshold) const {
  predictions.clear();
  if (in.peek() == EOF) {
    return false;
  }

  std::vector<int32_t> words, labels;
  dict_->getLine(in, words, labels);
  Predictions linePredictions;
  predict(k, words, linePredictions, threshold);
  for (const auto& p : linePredictions) {
    predictions.push_back(
        std::make_pair(std::exp(p.first), dict_->getLabel(p.second)));
  }

  return true;
}

void FastText::getSentenceVector(std::istream& in, fasttext::Vector& svec) {
  svec.zero();
  if (args_->model == model_name::sup) {
    std::vector<int32_t> line, labels;
    dict_->getLine(in, line, labels);
    for (int32_t i = 0; i < line.size(); i++) {
      addInputVector(svec, line[i]);
    }
    if (!line.empty()) {
      svec.mul(1.0 / line.size());
    }
  } else {
    Vector vec(args_->dim);
    std::string sentence;
    std::getline(in, sentence);
    std::istringstream iss(sentence);
    std::string word;
    int32_t count = 0;
    while (iss >> word) {
      getWordVector(vec, word);
      real norm = vec.norm();
      if (norm > 0) {
        vec.mul(1.0 / norm);
        svec.addVector(vec);
        count++;
      }
    }
    if (count > 0) {
      svec.mul(1.0 / count);
    }
  }
}

std::vector<std::pair<std::string, Vector>> FastText::getNgramVectors(
    const std::string& word) const {
  std::vector<std::pair<std::string, Vector>> result;
  std::vector<int32_t> ngrams;
  std::vector<std::string> substrings;
  dict_->getSubwords(word, ngrams, substrings);
  assert(ngrams.size() <= substrings.size());
  for (int32_t i = 0; i < ngrams.size(); i++) {
    Vector vec(args_->dim);
    if (ngrams[i] >= 0) {
      vec.addRow(*input_, ngrams[i]);
    }
    result.push_back(std::make_pair(substrings[i], std::move(vec)));
  }
  return result;
}

// deprecated. use getNgramVectors instead
void FastText::ngramVectors(std::string word) {
  std::vector<std::pair<std::string, Vector>> ngramVectors =
      getNgramVectors(word);

  for (const auto& ngramVector : ngramVectors) {
    std::cout << ngramVector.first << " " << ngramVector.second << std::endl;
  }
}

void FastText::precomputeWordVectors(DenseMatrix& wordVectors) {
  Vector vec(args_->dim);
  wordVectors.zero();
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    real norm = vec.norm();
    if (norm > 0) {
      wordVectors.addVectorToRow(vec, i, 1.0 / norm);
    }
  }
}

void FastText::lazyComputeWordVectors() {
  if (!wordVectors_) {
    wordVectors_ = std::unique_ptr<DenseMatrix>(
        new DenseMatrix(dict_->nwords(), args_->dim));
    precomputeWordVectors(*wordVectors_);
  }
}

std::vector<std::pair<real, std::string>> FastText::getNN(
    const std::string& word,
    int32_t k) {
  Vector query(args_->dim);

  getWordVector(query, word);

  lazyComputeWordVectors();
  assert(wordVectors_);
  return getNN(*wordVectors_, query, k, {word});
}

std::vector<std::pair<real, std::string>> FastText::getNN(
    const DenseMatrix& wordVectors,
    const Vector& query,
    int32_t k,
    const std::set<std::string>& banSet) {
  std::vector<std::pair<real, std::string>> heap;

  real queryNorm = query.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }

  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    if (banSet.find(word) == banSet.end()) {
      real dp = wordVectors.dotRow(query, i);
      real similarity = dp / queryNorm;

      if (heap.size() == k && similarity < heap.front().first) {
        continue;
      }
      heap.push_back(std::make_pair(similarity, word));
      std::push_heap(heap.begin(), heap.end(), comparePairs);
      if (heap.size() > k) {
        std::pop_heap(heap.begin(), heap.end(), comparePairs);
        heap.pop_back();
      }
    }
  }
  std::sort_heap(heap.begin(), heap.end(), comparePairs);

  return heap;
}

// depracted. use getNN instead
void FastText::findNN(
    const DenseMatrix& wordVectors,
    const Vector& query,
    int32_t k,
    const std::set<std::string>& banSet,
    std::vector<std::pair<real, std::string>>& results) {
  results.clear();
  results = getNN(wordVectors, query, k, banSet);
}

std::vector<std::pair<real, std::string>> FastText::getAnalogies(
    int32_t k,
    const std::string& wordA,
    const std::string& wordB,
    const std::string& wordC) {
  Vector query = Vector(args_->dim);
  query.zero();

  Vector buffer(args_->dim);
  getWordVector(buffer, wordA);
  query.addVector(buffer, 1.0 / (buffer.norm() + 1e-8));
  getWordVector(buffer, wordB);
  query.addVector(buffer, -1.0 / (buffer.norm() + 1e-8));
  getWordVector(buffer, wordC);
  query.addVector(buffer, 1.0 / (buffer.norm() + 1e-8));

  lazyComputeWordVectors();
  assert(wordVectors_);
  return getNN(*wordVectors_, query, k, {wordA, wordB, wordC});
}

std::vector<std::pair<real, std::string>> FastText::getAnalogies(
    int32_t k,
    const std::string& wordA,
    const std::string& wordB,
    const std::string& wordC,
    const std::string& wordR) {
  Vector query = Vector(args_->dim);
  query.zero();

  Vector buffer(args_->dim);
  getWordVector(buffer, wordA);
  query.addVector(buffer, 1.0 / (buffer.norm() + 1e-8));
  getWordVector(buffer, wordB);
  query.addVector(buffer, -1.0 / (buffer.norm() + 1e-8));
  getWordVector(buffer, wordC);
  query.addVector(buffer, 1.0 / (buffer.norm() + 1e-8));

  lazyComputeWordVectors();
  assert(wordVectors_);
  const std::set<std::string> banSet = {wordA, wordB, wordC};
  
  std::vector<std::pair<real, std::string>> heap;

  real queryNorm = query.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }

  std::string word = wordR;
  if (banSet.find(word) == banSet.end()) {
    bool f = false;
    for (int32_t i = 0; i < dict_->nwords(); i++) {
      if(!dict_->getWord(i).compare(word)) {
        real dp = wordVectors_->dotRow(query, i);
        real similarity = dp / queryNorm;
        heap.push_back(std::make_pair(similarity, word));
        f = true;
        break;
      }
    }
    if(!f)
      heap.push_back(std::make_pair(0.0, "nopeword"));
  }
  return heap;
}

// depreacted, use getAnalogies instead
void FastText::analogies(int32_t k) {
  std::string prompt("Query triplet (A - B + C)? ");
  std::string wordA, wordB, wordC;
  std::cout << prompt;
  while (true) {
    std::cin >> wordA;
    std::cin >> wordB;
    std::cin >> wordC;
    auto results = getAnalogies(k, wordA, wordB, wordC);

    for (auto& pair : results) {
      std::cout << pair.second << " " << pair.first << std::endl;
    }
    std::cout << prompt;
  }
}

void FastText::trainThread(int32_t threadId) {
  std::ifstream ifs(args_->input);
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model::State state(args_->dim, output_->size(0), threadId);
  
  std::vector<int32_t> gradMap(2 * args_->ws + 1, -1);
  std::vector<Vector> gradMat(2 * args_->ws + 1, Vector(args_->dim));

  const int64_t ntokens = dict_->ntokens();
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels;
  while (tokenCount_ < args_->epoch * ntokens) {
    real progress = real(tokenCount_) / (args_->epoch * ntokens);
    real lr = args_->lr * (1.0 - progress);
    if (args_->model == model_name::sup) {
      localTokenCount += dict_->getLine(ifs, line, labels);
      supervised(state, lr, line, labels);
    } else if (args_->model == model_name::cbow) {
      localTokenCount += dict_->getLine(ifs, line, state.rng);
      switch(args_->mode) {
        case mode_name::normal:
          cbow(state, lr, line);
          break;
        case mode_name::ns:
          cbow_shared(threadId, state, lr, line);
          break;
        case mode_name::uphid:
         cbow_uphid(threadId, state, lr, line, gradMap, gradMat);
         break;
        case mode_name::uphid_ns:
          cbow_uphid_shared(threadId, state, lr, line, gradMap, gradMat);
          break;
        case mode_name::uphid_fixed:
          cbow_uphid_fixed(threadId, state, lr, line, gradMap, gradMat);
          break;
        case mode_name::uphid_ns_fixed:
         cbow_uphid_shared_fixed(threadId, state, lr, line, gradMap, gradMat);
        break;
      }
    } else if (args_->model == model_name::sg) {
      localTokenCount += dict_->getLine(ifs, line, state.rng);
      switch(args_->mode) {
        case mode_name::normal:
          skipgram(state, lr, line);
          break;
        case mode_name::batched:
          skipgram_batched(threadId, state, lr, line);
          break;
        case mode_name::ns:
          skipgram_ns(threadId, state, lr, line);
          break;
      }
    }
    if (localTokenCount > args_->lrUpdateRate) {
      tokenCount_ += localTokenCount;
      localTokenCount = 0;
      if (threadId == 0 && args_->verbose > 1)
        loss_ = state.getLoss();
    }
  }
  if (threadId == 0)
    loss_ = state.getLoss();
  ifs.close();
}

std::shared_ptr<Matrix> FastText::getInputMatrixFromFile(
    const std::string& filename) const {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<DenseMatrix> mat; // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  in >> n >> dim;
  if (dim != args_->dim) {
    throw std::invalid_argument(
        "Dimension of pretrained vectors (" + std::to_string(dim) +
        ") does not match dimension (" + std::to_string(args_->dim) + ")!");
  }
  #if STORE_IN_AVX
    std::cout << "FATAL PROGRAMMER ERROR: READING INPUT MATRIX FROM FILE IS NOT IMPLEMENTED YET\n";
    exit(-1);
  #else
  mat = std::make_shared<DenseMatrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->add(word);
    for (size_t j = 0; j < dim; j++) {
      in >> mat->at(i, j);
    }
  }
  #endif
  in.close();

  dict_->threshold(1, 0);
  dict_->init();
  std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
      dict_->nwords() + args_->bucket, args_->dim);
  input->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++) {
    int32_t idx = dict_->getId(words[i]);
    if (idx < 0 || idx >= dict_->nwords()) {
      continue;
    }
    for (size_t j = 0; j < dim; j++) {
      input->at(idx, j) = mat->at(i, j);
    }
  }
  return input;
}

void FastText::loadVectors(const std::string& filename) {
  input_ = getInputMatrixFromFile(filename);
}

std::shared_ptr<Matrix> FastText::createRandomMatrix() const {
  std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
      dict_->nwords() + args_->bucket, args_->dim);
  input->uniform(1.0 / args_->dim);

  return input;
}

std::shared_ptr<Matrix> FastText::createTrainOutputMatrix() const {
  int64_t m =
      (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
  std::shared_ptr<DenseMatrix> output =
      std::make_shared<DenseMatrix>(m, args_->dim);
  output->zero();

  return output;
}

void FastText::train(const Args& args) {
  args_ = std::make_shared<Args>(args);
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    throw std::invalid_argument("Cannot use stdin for training!");
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    throw std::invalid_argument(
        args_->input + " cannot be opened for training!");
  }
  if (args_->tokenVocab == "") {
    dict_->readFromFile(ifs);
  } else {
    std::ifstream vifs(args_->tokenVocab);
    if (!vifs.is_open()) {
      throw std::invalid_argument(
          args_->tokenVocab + " cannot be opened for training!");
    }
    
    std::ifstream mifs(args_->tokenMerges);
    if (!mifs.is_open()) {
      throw std::invalid_argument(
          args_->tokenMerges + " cannot be opened for training!");
    }
    dict_->readFromFileTokenized(ifs, vifs, mifs);
    mifs.close();
    vifs.close();
  }
  ifs.close();
  
  if (!args_->pretrainedVectors.empty()) {
    input_ = getInputMatrixFromFile(args_->pretrainedVectors);
  } else {
    input_ = createRandomMatrix();
  }
  output_ = createTrainOutputMatrix();
  auto loss = createLoss(output_);
  bool normalizeGradient = (args_->model == model_name::sup);
  model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
  std::cout << "Training...\n";
  auto start_time = std::chrono::high_resolution_clock::now(); 
  startThreads();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed =  (std::chrono::duration<double>(end_time - start_time)).count();
  std::cout << elapsed << std::endl;
}

void FastText::startThreads() {
  start_ = std::chrono::steady_clock::now();
  tokenCount_ = 0;
  loss_ = -1;
  std::vector<std::thread> threads;
  for (int32_t i = 0; i < args_->thread; i++) {
    threads.push_back(std::thread([=]() { trainThread(i); }));
  }
  const int64_t ntokens = dict_->ntokens();
  // Same condition as trainThread
  while (tokenCount_ < args_->epoch * ntokens) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //if (loss_ >= 0 && args_->verbose > 1) {
      real progress = real(tokenCount_) / (args_->epoch * ntokens);
      std::cerr << "\r";
      printInfo(progress, loss_, std::cerr);
    //}
  }
  for (int32_t i = 0; i < args_->thread; i++) {
    threads[i].join();
  }
  if (args_->verbose > 0) {
    std::cerr << "\r";
    printInfo(1.0, loss_, std::cerr);
    std::cerr << std::endl;
  }
}

int FastText::getDimension() const {
  return args_->dim;
}

bool FastText::isQuant() const {
  return quant_;
}

bool comparePairs(
    const std::pair<real, std::string>& l,
    const std::pair<real, std::string>& r) {
  return l.first > r.first;
}

} // namespace fasttext
