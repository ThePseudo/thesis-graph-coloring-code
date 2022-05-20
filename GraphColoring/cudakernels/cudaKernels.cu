#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/count.h>

#include "cudaKernels.h"
#include "cusparse.h"

#include <algorithm>
#include <iostream>
#include <vector>

#define CUDA_MAX_BLOCKS 2147483647 // Maximum blocks to launch, depending on GPU

#define CUDA_SAFE_CALL(ans) { cudaSafeCheck((ans), __FILE__, __LINE__);}
inline void cudaSafeCheck(cudaError_t call, const char *file, int line, bool abort=true){
  if (call != cudaSuccess){
    printf("Error: %s in file: %s at line: %d\n", cudaGetErrorString(call), file, line);
    if (abort)
      exit(call);
  }
}

__global__ void create_independent_set_kernel(int n, const int* Ao, const int* Ac, const int* randoms, const int* colors, unsigned int* set);
__global__ void expand_to_maximal_independent_set_kernel(int n, const int* Ao, const int* Ac, const int* colors, unsigned int* set);
__global__ void color_jpl_kernel(int n, int c, int* colors, const unsigned int* set);

int color_jpl(int const n, const int* Ao, const int* Ac, int* colors, const int* randoms) {
	int* dAo;
	int* dAc;
	int* dRandoms;
	int* dColors;
	unsigned int* dSet;
	Benchmark& bm = *Benchmark::getInstance();

	CUDA_SAFE_CALL(cudaMalloc(&dAo, (n + 1) * sizeof(*dAo)));
	CUDA_SAFE_CALL(cudaMalloc(&dAc, Ao[n] * sizeof(*dAc)));
	CUDA_SAFE_CALL(cudaMalloc(&dRandoms, n * sizeof(*dRandoms)));
	CUDA_SAFE_CALL(cudaMalloc(&dColors, n * sizeof(*dColors)));
	CUDA_SAFE_CALL(cudaMalloc(&dSet, n * sizeof(*dSet)));

	CUDA_SAFE_CALL(cudaMemcpy(dAo, Ao, (n + 1) * sizeof(*Ao), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dAc, Ac, Ao[n] * sizeof(*Ac), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dRandoms, randoms, n * sizeof(*randoms), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dColors, colors, n * sizeof(*colors), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemset(dSet, 0x0, n * sizeof(*dSet)));

	bm.sampleTimeToFlag(2);

	int c;
	int left;
	int const nt = 256;
	int nb = std::min((n + nt - 1) / nt, CUDA_MAX_BLOCKS);
	for (c = 0, left = n; left > 0 && c < n; ++c) {
		bm.sampleTimeToFlag(4);

		create_independent_set_kernel<<<nb, nt>>>(n, dAo, dAc, dRandoms, dColors, dSet);
		//expand_to_maximal_independent_set_kernel<<<nb, nt>>>(n, dAo, dAc, dColors, dSet);
		color_jpl_kernel<<<nb, nt>>>(n, c, dColors, dSet);

		cudaDeviceSynchronize();
		bm.sampleTimeToFlag(1);

		CUDA_SAFE_CALL(cudaMemcpy(colors, dColors, n * sizeof(*colors), cudaMemcpyDeviceToHost));
		bm.sampleTimeToFlag(3);

		left = (int)thrust::count(colors, colors + n, -1);
	}

	CUDA_SAFE_CALL(cudaFree(dAo));
	CUDA_SAFE_CALL(cudaFree(dAc));
	CUDA_SAFE_CALL(cudaFree(dRandoms));
	CUDA_SAFE_CALL(cudaFree(dColors));
	CUDA_SAFE_CALL(cudaFree(dSet));

	return c;
}

__global__ void create_independent_set_kernel(int n, const int* Ao, const int* Ac, const int* randoms, const int* colors, unsigned int* set) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		i < n;
		i += blockDim.x * gridDim.x)
	{
		//bool f = true; // true if you have max random

		// ignore nodes colored earlier
		if (colors[i] != -1) continue;
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
		set[i] = 0;
#endif
#ifdef COLOR_MAX_INDEPENDENT_SET
		set[i] = 1;
#endif
		int ir = randoms[i];

		// look at neighbors to check their random number
		for (int k = Ao[i]; k < Ao[i + 1]; k++) {
			// ignore nodes colored earlier (and yourself)
			int j = Ac[k];
			int jc = colors[j];
			if ((jc != -1) || (i == j)) continue;
			int jr = randoms[j];
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
			if (set[i] == 0 && ir <= jr) set[i] = 1;
			else if (set[i] == 0 && ir > jr) set[i] = 2;
			else if (set[i] == 1 && ir > jr) set[i] = 3;
			else if (set[i] == 2 && ir <= jr) set[i] = 3;
#endif
#ifdef COLOR_MAX_INDEPENDENT_SET
			if (ir <= jr) set[i] = 0;
#endif
		}
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
		if (set[i] == 0) set[i] = 1;
#endif
		// assign color if you have the maximum random number
		//if (f) colors[i] = c;
	}
}

__global__ void expand_to_maximal_independent_set_kernel(int n, const int* Ao, const int* Ac, const int* colors, unsigned int* set) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		i < n;
		i += blockDim.x * gridDim.x)
	{
		// Ignore nodes colored earlier or already in set
		if (colors[i] != -1 || set[i] != 0x0) continue;

		set[i] = 0x2;

		for (int k = Ao[i]; k < Ao[i + 1]; k++) {
			// ignore nodes colored earlier (and yourself)
			int j = Ac[k];
			int jc = colors[j];
			if ((jc != -1) || (i == j)) continue;
			// cannot be part of MIS if neighbor is in initial set
			//  or if neighboring vertex with higher degree is trying to enter the MIS
			if (set[j] == 0x1 ||
				(set[j] == 0x2 && Ao[i + 1] - Ao[i] <= Ao[j + 1] - Ao[j])
				)
				set[i] = 0x0;
		}

		if (set[i] != 0x0) set[i] = 0x1;
	}
}

__global__ void color_jpl_kernel(int n, int c, int* colors, const unsigned int* set) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		i < n;
		i += blockDim.x * gridDim.x)
	{
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
		if (colors[i] != -1 || set[i] == 0 || set[i] == 3) continue;
		colors[i] = 2 * c + set[i] - 1;
#endif
#ifdef COLOR_MAX_INDEPENDENT_SET
		if (colors[i] != -1) continue;
		if(set[i] != 0) colors[i] = c;
#endif
	}
}

int color_cusparse(int const n, const int* Ao, const int* Ac, int* colors) {
	float* dAv;
	int* dAo;
	int* dAc;
	int* dColors;

	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();

	CUDA_SAFE_CALL(cudaMalloc(&dAo, (n + 1) * sizeof(*dAo)));
	CUDA_SAFE_CALL(cudaMemcpy(dAo, Ao, (n + 1) * sizeof(*Ao), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc(&dAc, Ao[n] * sizeof(*dAc)));
	CUDA_SAFE_CALL(cudaMemcpy(dAc, Ac, Ao[n] * sizeof(*Ac), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc(&dColors, n * sizeof(*dColors)));
	CUDA_SAFE_CALL(cudaMemcpy(dColors, colors, n * sizeof(*colors), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc(&dAv, Ao[n] * sizeof(*dAv)));

	int c;
	float fractionToColor = 1.0;

	cusparseStatus_t status;
	cusparseHandle_t handle;
	cusparseMatDescr_t matrixDesc;
	cusparseColorInfo_t colorInfo;

	status = cusparseCreate(&handle);
	status = cusparseCreateMatDescr(&matrixDesc);
	status = cusparseCreateColorInfo(&colorInfo);


	bm.sampleTimeToFlag(1);
	status = cusparseScsrcolor(handle,
		n,
		Ao[n],
		matrixDesc,
		dAv,
		dAo,
		dAc,
		&fractionToColor,
		&c,
		dColors,
		NULL,
		colorInfo);

	cudaDeviceSynchronize();
	bm.sampleTimeToFlag(2);

	CUDA_SAFE_CALL(cudaMemcpy(colors, dColors, n * sizeof(*colors), cudaMemcpyDeviceToHost));

	bm.sampleTimeToFlag(3);
	CUDA_SAFE_CALL(cudaFree(dAv));
	CUDA_SAFE_CALL(cudaFree(dAo));
	CUDA_SAFE_CALL(cudaFree(dAc));
	CUDA_SAFE_CALL(cudaFree(dColors));

	cusparseDestroyMatDescr(matrixDesc);
	cusparseDestroyColorInfo(colorInfo);
	cusparseDestroy(handle);

	return c;
}