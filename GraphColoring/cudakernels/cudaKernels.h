#ifndef _CUDA_KERNELS_H
#define _CUDA_KERNELS_H

// #include "configuration.h"
// #include "benchmark.h"
#include <iostream>
#include <vector>

struct NewBenchmark {
  double ms_per_randomization = 0;
  double ms_allocation = 0;
  double ms_transfer_to_gpu = 0;
  double ms_execute = 0;
  double ms_transfer_to_cpu = 0;
  double ms_total_process = 0;
  std::vector<std::vector<uint64_t>> colored;

  static NewBenchmark &get() {
    static NewBenchmark bm = NewBenchmark();
    return bm;
  }

  double getTotaltime() {
    return ms_per_randomization + ms_allocation + ms_transfer_to_gpu +
           ms_execute + ms_transfer_to_cpu;
  }
};

int color_jpl(int const n, const int *Ao, const int *Ac, int *colors,
              const int *randoms, int resetCount);
int color_cusparse(int const n, const int *Ao, const int *Ac, int *colors,
                   int resetCount);

#endif // !_CUDA_KERNELS_H
