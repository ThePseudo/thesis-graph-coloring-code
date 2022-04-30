#ifndef _CUDA_KERNELS_H
#define _CUDA_KERNELS_H

#include "configuration.h"
#include "benchmark.h"

int color_jpl(int const n, const int* Ao, const int* Ac, int* colors, const int* randoms);
int color_cusparse(int const n, const int* Ao, const int* Ac, int* colors);
#endif // !_CUDA_KERNELS_H