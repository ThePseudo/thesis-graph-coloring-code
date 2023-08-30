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
                  std::shared_ptr<kp::TensorT<int>> tAo,
                  std::shared_ptr<kp::TensorT<int>> tAc,
                  std::shared_ptr<kp::TensorT<int>> tRandoms,
                  std::shared_ptr<kp::TensorT<int>> tColors,
                  std::shared_ptr<kp::TensorT<int>> tFinished,
                  std::vector<uint32_t> &shader) {
  uint32_t nt = 256; // Number of threads per block to be launched
  uint32_t nb = (last - first + nt - 1) / nt; // Number of blocks to be launched

  std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
      {tAo, tAc, tRandoms, tColors, tFinished}, {shader}, {}, {}, {0, 0, 0});
  int c = -1;
  bool finished = false;
  int left = last - first;
  for (c = 0; !finished; c++) {
    tFinished->setData({1});
    algo->setPushConstants<int32_t>({first, last, c});
    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>({tFinished})
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpTensorSyncLocal>({tFinished})
        ->eval();
    finished = tFinished->data()[0];
    // std::cout << finished << std::endl;
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
  auto tAo = mgr.tensorT(std::vector<int>(Ao - first, Ao + last - first - 1));
  auto tAc = mgr.tensorT(std::vector<int>(Ac - first, Ac + last - first - 1));
  auto tRandoms = mgr.tensorT(
      std::vector<int>(randoms - first, randoms + last - first - 1));
  auto tColors =
      mgr.tensorT(std::vector<int>(colors - first, colors - first + last - 1));
  auto tFinished = mgr.tensorT<int>({1});
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
  return needConflictCheck ? -c : c;
}
