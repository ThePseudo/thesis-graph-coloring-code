#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/count.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cooperative_groups.h>


#include "cudaKernels.h"
#include "cusparse.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

#define CUDA_SAFE_CALL(ans) { cudaSafeCheck((ans), __FILE__, __LINE__);}
inline void cudaSafeCheck(cudaError_t call, const char *file, int line, bool abort=true){
  if (call != cudaSuccess){
    printf("Error: %s in file: %s at line: %d\n", cudaGetErrorString(call), file, line);
    if (abort)
      exit(call);
  }
}

int launch_kernel(int n, int* dAo, int* dAc, int* dRandoms, int* dColors, unsigned int* dSet, int* colors);
int launch_kernel_coop(int n, const int* dAo, const int* dAc, const int* dRandoms, int* dColors, int* colors);

__global__ void create_independent_set_kernel(int n, const int* Ao, const int* Ac, const int* randoms, const int* colors, unsigned int* set);
__global__ void expand_to_maximal_independent_set_kernel(int n, const int* Ao, const int* Ac, const int* colors, unsigned int* set);
__global__ void color_jpl_kernel(int n, int c, int* colors, const unsigned int* set);
__global__ void color_jpl_coop(int n, const int* Ao, const int* Ac, const int* randoms, int* colors);

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

	int device = 0;
	int supportsCoopLaunch = 0;
	cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);

	int c = -1;
	if (supportsCoopLaunch) {
		std::cout << "Launching in cooperative mode🤝" << std::endl;
		c = launch_kernel_coop(n, dAo, dAc, dRandoms, dColors, colors);
	} else {
		c = launch_kernel(n, dAo, dAc, dRandoms, dColors, dSet, colors);
	}

	CUDA_SAFE_CALL(cudaFree(dAo));
	CUDA_SAFE_CALL(cudaFree(dAc));
	CUDA_SAFE_CALL(cudaFree(dRandoms));
	CUDA_SAFE_CALL(cudaFree(dColors));
	CUDA_SAFE_CALL(cudaFree(dSet));

	return c;
}

int launch_kernel(int n, int* dAo, int* dAc, int* dRandoms, int* dColors, unsigned int* dSet, int* colors) {
	int device = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	Benchmark& bm = *Benchmark::getInstance();
	int c = -1;
	int left;
	
	int const nt = 256;
	int nb = std::min((n + nt - 1) / nt, deviceProp.maxGridSize[0]);

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

	return c;
}

int launch_kernel_coop(int n, const int* dAo, const int* dAc, const int* dRandoms, int* dColors, int* colors) {
	int c = -1;
	int device = 0;
	int nb = 0;
	int nt = 256;
	Benchmark& bm = *Benchmark::getInstance();

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, color_jpl_coop, nt, 0);
	nb = std::min((n + nt - 1) / nt, nb);

	void* kernelArgs[] = {&n, &dAo, &dAc, &dRandoms, &dColors};
	dim3 dimBlock(nt, 1, 1);
	dim3 dimGrid(deviceProp.multiProcessorCount * nb, 1, 1);
	cudaLaunchCooperativeKernel((void*)color_jpl_coop, dimGrid, dimBlock, kernelArgs);

	cudaDeviceSynchronize();
	bm.sampleTimeToFlag(1);

	CUDA_SAFE_CALL(cudaMemcpy(colors, dColors, n * sizeof(*colors), cudaMemcpyDeviceToHost));
	bm.sampleTimeToFlag(3);

	c = *std::max_element(colors, colors+n);
	bm.sampleTimeToFlag(4);

	return c;
}

__global__ void color_jpl_coop(int n, const int* Ao, const int* Ac, const int* randoms, int* colors) {
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	int left = 1;

	for (int c = 0; c < n && left > 0; ++c) {
		left = 0;
		for (int i = threadIdx.x + blockIdx.x * blockDim.x;
			i < n;
			i += blockDim.x * gridDim.x)
		{
			//bool f = true; // true if you have max random

		// ignore nodes colored earlier
			if (colors[i] != -1) continue;
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
			bool localmin = true;
#endif
			bool localmax = true;
			int ir = randoms[i];

			// look at neighbors to check their random number
			for (int k = Ao[i]; k < Ao[i + 1]; k++) {
				// ignore nodes colored earlier (and yourself)
				int j = Ac[k];
				int jc = colors[j];
				if ((jc != -1) || (i == j)) continue;
				int jr = randoms[j];
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
				localmin &= ir < jr;
#endif
				localmax &= ir > jr;
			}
			// assign color if you have the maximum random number
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
			if (localmin) colors[i] = 2 * c + 1;
			if (localmax) colors[i] = 2 * c;
			if (!localmax && !localmin) ++left;
#endif
#ifdef COLOR_MAX_INDEPENDENT_SET
			if (localmax) colors[i] = c;
			else ++left;
#endif
		}
		grid.sync();
	}
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