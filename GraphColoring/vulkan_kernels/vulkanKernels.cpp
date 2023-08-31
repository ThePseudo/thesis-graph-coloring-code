#include "vulkanKernels.h"
#include "../NewBenchmark.h"
#include "kompute/Algorithm.hpp"
#include "kompute/Manager.hpp"
#include "kompute/Tensor.hpp"
#include "kompute/operations/OpAlgoDispatch.hpp"
#include "kompute/operations/OpTensorSyncDevice.hpp"
#include "kompute/operations/OpTensorSyncLocal.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

std::vector<uint32_t> loadShaderFromFile(const char *fileName) {
  using namespace std;
  std::vector<uint32_t> result;
  ifstream fin(fileName, ios::binary);
  if (fin) {
    auto start = fin.tellg();
    fin.seekg(0, ios::end);
    auto end = fin.tellg();
    fin.seekg(0, ios::beg);
    auto size = end - start;
    result.resize(size / sizeof(uint32_t), 0);
    fin.read((char *)result.data(), size);
  } else {
    throw std::runtime_error("Could not find shader: " + std::string(fileName));
  }
  return result;
}

int launch_kernel(const int first, const int last, kp::Manager &mgr,
                  std::shared_ptr<kp::TensorT<int32_t>> tAo,
                  std::shared_ptr<kp::TensorT<int32_t>> tAc,
                  std::shared_ptr<kp::TensorT<int32_t>> tRandoms,
                  std::shared_ptr<kp::TensorT<int32_t>> tColors,
                  std::shared_ptr<kp::TensorT<int32_t>> tFinished,
                  std::vector<uint32_t> &shader) {
  uint32_t nt = last - first; // Number of threads per block to be launched
  uint32_t nb = (last - first + nt - 1) / nt; // Number of blocks to be launched
  nb = 1;
  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm({tAo, tAc, tRandoms, tColors, tFinished}, {shader},
                    {nt, nb}, {}, {0, 0, 0, 0});
  int c = -1;
  bool finished = false;
  int left = last - first;
  tFinished->setData({1});
  auto seq = mgr.sequence()
                 ->record<kp::OpTensorSyncDevice>({tFinished})
                 ->record<kp::OpAlgoDispatch>(
                     algo, std::vector<int32_t>{static_cast<int32_t>(nt), first,
                                                last, c})
                 ->record<kp::OpTensorSyncLocal>({tFinished});
  for (c = 0; !finished; c++) {
    seq->evalAsync();
    auto seq1 = mgr.sequence()
                    ->record<kp::OpTensorSyncDevice>({tFinished})
                    ->record<kp::OpAlgoDispatch>(
                        algo, std::vector<int32_t>{static_cast<int32_t>(nt),
                                                   first, last, c})
                    ->record<kp::OpTensorSyncLocal>({tFinished});
    seq->evalAwait();
    finished = tFinished->data()[0];
    tFinished->setData({1});
    seq = seq1;
  }
  return c;
}

int color_jpl(int const n, const int *Ao, const int *Ac, int *colors,
              const int *randoms, int resetCount) {
  int c = -1;
  int first = 0;
  int last = n;
  bool needConflictCheck = false;
  auto start_proc = std::chrono::high_resolution_clock::now();
  kp::Manager mgr;
  auto shader = loadShaderFromFile("shaders/coloring.comp.spv");
  if (shader.empty()) {
    throw std::runtime_error("Shader empty!");
  }

  auto start = std::chrono::high_resolution_clock::now();
  auto tAo =
      mgr.tensorT(std::vector<int32_t>(Ao - first, Ao + last - first + 1));
  auto tAc = mgr.tensorT(
      std::vector<int32_t>(Ac + Ao[first], Ac + Ao[last] - Ao[first]));
  auto tRandoms = mgr.tensorT(
      std::vector<int32_t>(randoms - first, randoms + last - first));
  auto tColors =
      mgr.tensorT(std::vector<int32_t>(colors - first, colors - first + last));
  auto tFinished = mgr.tensorT<int32_t>({1});
  auto end = std::chrono::high_resolution_clock::now();
  NewBenchmark::get().ms_allocation +=
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0f;

  start = std::chrono::high_resolution_clock::now();
  mgr.sequence()
      ->record<kp::OpTensorSyncDevice>({tAo, tAc, tRandoms, tColors})
      ->eval();
  end = std::chrono::high_resolution_clock::now();
  NewBenchmark::get().ms_transfer_to_gpu +=
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0f;
  start = std::chrono::high_resolution_clock::now();
  c += launch_kernel(first, last, mgr, tAo, tAc, tRandoms, tColors, tFinished,
                     shader);
  end = std::chrono::high_resolution_clock::now();
  NewBenchmark::get().ms_execute +=
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0f;
  start = std::chrono::high_resolution_clock::now();
  mgr.sequence()->record<kp::OpTensorSyncLocal>({tColors})->eval();
  end = std::chrono::high_resolution_clock::now();
  NewBenchmark::get().ms_transfer_to_cpu +=
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0f;
  memcpy(colors, tColors->data(), sizeof(colors[0]) * tColors->size());
  return needConflictCheck ? -c : c;
}
