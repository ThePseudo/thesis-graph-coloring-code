#pragma once

struct NewBenchmark {
  double ms_per_randomization = 0;
  double ms_allocation = 0;
  double ms_transfer_to_gpu = 0;
  double ms_execute = 0;
  double ms_transfer_to_cpu = 0;
  double ms_total_process = 0;

  static NewBenchmark &get() {
    static NewBenchmark bm = NewBenchmark();
    return bm;
  }

  double getTotaltime() {
    return ms_per_randomization + ms_allocation + ms_transfer_to_gpu +
           ms_execute + ms_transfer_to_cpu;
  }
};
