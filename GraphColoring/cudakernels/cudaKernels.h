#ifndef _CUDA_KERNELS_H
#define _CUDA_KERNELS_H

// #include "configuration.h"
#include "../NewBenchmark.h"

int color_jpl(int const n, const int *Ao, const int *Ac, int *colors,
              const int *randoms, int resetCount);
int color_cusparse(int const n, const int *Ao, const int *Ac, int *colors,
                   int resetCount);

#endif // !_CUDA_KERNELS_H
