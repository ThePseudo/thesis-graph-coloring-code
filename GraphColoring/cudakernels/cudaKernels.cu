﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//#include <cuda_runtime_api.h>
//#include <cuda.h>
//#include <cooperative_groups.h>

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

float checkDeviceMemOccupancy(size_t required) {
	int device = 0;
	size_t totalGlobalMemory = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	totalGlobalMemory = prop.totalGlobalMem;

	return required * 1.0f / totalGlobalMemory;
}

size_t calc_jpl_memory_occupancy(int const first, int const last, const int* Ao) {
	return sizeof(*Ao) * (3 * (last - first) + Ao[last] - Ao[first] + 1);
}

size_t calc_cusparse_memory_occupancy(int const first, int const last, const int* Ao) {
	return sizeof(*Ao) * (2 * (last - first) + Ao[last] - Ao[first] + 1) +
		sizeof(float) * (Ao[last] - Ao[first]);
}

int launch_kernel(int const first, int const last, int* dAo, int* dAc, int* dRandoms, thrust::device_vector<int>& dvColors);
__global__ void color_jpl_kernel(const int first, const int last, const int c, const int* Ao, const int* Ac, const int* randoms, int* colors);

//int launch_kernel_coop(int n, const int* dAo, const int* dAc, const int* dRandoms, int* dColors, int* colors);
//__global__ void color_jpl_coop_kernel(int n, const int* Ao, const int* Ac, const int* randoms, int* colors);

__device__ bool color_jpl_ingore_neighbor(const int c, const int i, const int j, const int jc);
__device__ bool color_jpl_assign_color(const int c, int* color_i, const bool localmax, const bool localmin = false);

int color_jpl(int const n, const int* Ao, const int* Ac, int* colors, const int* randoms, int resetCount) {
	int c = -1;
	int* dAo;
	int* dAc;
	int* dRandoms;
	int first;
	int last;
	bool needConflictCheck = false;

	Benchmark& bm = *Benchmark::getInstance(resetCount);
	bm.sampleTime();

	for (first = 0, last = n; first < n; first = last, last = n) {
		size_t occupancyBytes;
		float occupancyPerc;
		do {
			occupancyBytes = calc_cusparse_memory_occupancy(first, last, Ao);
			occupancyPerc = checkDeviceMemOccupancy(occupancyBytes);
			if (occupancyPerc > 1.0) {
				//printf("Required %.2f%% of global memory.\n", occupancyPerc * 100);
				//std::cout << "Unavailable resources. Program will be stopped." << std::endl;
				//return -1;
				last = first + (int)((last - first) * 1.0f / occupancyPerc);
				needConflictCheck = true;
			}
			if (first == last) {
				return -1;	// Device cannot fit a node alone
			}
		} while (occupancyPerc >= 1.0f);

		thrust::device_vector<int> dvColors(last - first, -1);

		CUDA_SAFE_CALL(cudaMalloc(&dAo, (last - first + 1) * sizeof(*dAo)));
		CUDA_SAFE_CALL(cudaMalloc(&dAc, (Ao[last] - Ao[first]) * sizeof(*dAc)));
		CUDA_SAFE_CALL(cudaMalloc(&dRandoms, (last - first) * sizeof(*dRandoms)));

		CUDA_SAFE_CALL(cudaMemcpy(dAo, Ao + first, (last - first + 1) * sizeof(*Ao), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(dAc, Ac + Ao[first], (Ao[last] - Ao[first]) * sizeof(*Ac), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(dRandoms, randoms + first, (last - first) * sizeof(*randoms), cudaMemcpyHostToDevice));

		bm.sampleTimeToFlag(1);

		c += launch_kernel(first, last, dAo, dAc, dRandoms, dvColors);
		bm.sampleTimeToFlag(2);

		// Copy colors array from device
		thrust::copy(dvColors.begin(), dvColors.end(), colors + first);
		bm.sampleTimeToFlag(3);

		CUDA_SAFE_CALL(cudaFree(dAo));
		CUDA_SAFE_CALL(cudaFree(dAc));
		CUDA_SAFE_CALL(cudaFree(dRandoms));
	}

	return needConflictCheck ? -c : c;
}

int launch_kernel(int const first, int const last, int* dAo, int* dAc, int* dRandoms, thrust::device_vector<int>& dvColors) {
	int c = -1;	// Number of colors used
	int left = last - first;	// Number of non-colored vertices

	int nt = 256;	// Number of threads per block to be launched
	int nb = (last - first + nt - 1) / nt;	// Number of blocks to be launched
	// Limit blocks to be launched by the number of vertices (one thread per vertex)

	// Get optimal number of blocks and threads to launch to fill SMs
//	cudaOccupancyMaxPotentialBlockSize(&nb, &nt, color_jpl_kernel, 0, 0);
	

	// Get raw pointer to device array
	int* dColors = thrust::raw_pointer_cast(dvColors.data());
	for (c = 0; left > 0 && c < last - first; ++c) {
		// Launch coloring iteration kernel
		color_jpl_kernel<<<nb, nt>>>(first, last, c, dAo, dAc, dRandoms, dColors);
		//cudaDeviceSynchronize();	// Not necessary, but useful to categoryze berchmark 

		// Count non-colored vertices on device
		left = (int)thrust::count(dvColors.begin(), dvColors.end(), -1);
	}

	return c;
}

__global__ void color_jpl_kernel(const int first, const int last, const int c, const int* Ao, const int* Ac, const int* randoms, int* colors) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		i < last - first;
		i += blockDim.x * gridDim.x)
	{

		int color = c;
		// true if you have max random
		bool localmax = true;
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
		// true if you have min random
		bool localmin = true;
		color *= 2;
#endif

		// ignore nodes colored earlier
		if (colors[i] != -1) continue;

		// look at neighbors to check their random number
		for (int k = Ao[i]; k < Ao[i + 1]; k++) {
			int j = Ac[k - Ao[0]] - first;

			// ignore nodes colored earlier (and yourself)
			if (j < 0 || j >= last - first || color_jpl_ingore_neighbor(color, i, j, colors[j])) continue;

			int ir, jr;
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
			ir = randoms[(i + (color<<1)) % (last - first)];
			jr = randoms[(j + (color<<1)) % (last - first)];
#endif
#ifdef COLOR_MAX_INDEPENDENT_SET
			ir = randoms[(i + color) % (last - first)];
			jr = randoms[(j + color) % (last - first)];
#endif

			localmax &= ir > jr;
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
			localmin &= ir < jr;
#endif
		}
		// assign color if you have the maximum (or minimum) random number
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
		color_jpl_assign_color(color, &colors[i], localmax, localmin);
#elif defined(COLOR_MAX_INDEPENDENT_SET)
		color_jpl_assign_color(color, &colors[i], localmax);
#endif
	}
}

/*****************************************************************************************************
* color_jpl implementation with Cooperative Groups (CUDA >= 9.0, CC >= 6.0)
* Seems to not be working
int launch_kernel_coop(int n, const int* dAo, const int* dAc, const int* dRandoms, int* dColors, int* colors) {
	int c = -1;
	int device = 0;
	int nb;
	int nt;
	Benchmark& bm = *Benchmark::getInstance();

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, color_jpl_coop, nt, 0);
	//nb = std::min((n + nt - 1) / nt, nb);
	cudaOccupancyMaxPotentialBlockSize(&nb, &nt, color_jpl_coop_kernel, 0, 0);

	void* kernelArgs[] = {&n, &dAo, &dAc, &dRandoms, &dColors};
	dim3 dimBlock(nt, 1, 1);
	dim3 dimGrid(nb, 1, 1);
	cudaLaunchCooperativeKernel((void*)color_jpl_coop_kernel, dimGrid, dimBlock, kernelArgs);

	cudaDeviceSynchronize();
	bm.sampleTimeToFlag(1);

	CUDA_SAFE_CALL(cudaMemcpy(colors, dColors, n * sizeof(*colors), cudaMemcpyDeviceToHost));
	bm.sampleTimeToFlag(3);

	c = *std::max_element(colors, colors+n);
	bm.sampleTimeToFlag(4);

	return c;
}

__global__ void color_jpl_coop_kernel(int n, const int* Ao, const int* Ac, const int* randoms, int* colors) {
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	int left = 1;

	for (int c = 0; c < n && left > 0; ++c) {
		left = 0;
		for (int i = threadIdx.x + blockIdx.x * blockDim.x;
			i < n;
			i += blockDim.x * gridDim.x)
		{

			int color = c;
			// true if you have max random
			bool localmax = true;
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
			// true if you have min random
			bool localmin = true;
			color *= 2;
#endif

			// ignore nodes colored earlier
			if (colors[i] != -1) continue;

			// look at neighbors to check their random number
			for (int k = Ao[i]; k < Ao[i + 1]; k++) {
				// ignore nodes colored earlier (and yourself)
				int j = Ac[k];

				if (color_jpl_ingore_neighbor(color, i, j, colors[j])) continue;

				int ir = randoms[i];
				int jr = randoms[j];

				localmax &= ir > jr;
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
				localmin &= ir < jr;
#endif
			}
			// assign color if you have the maximum (or minimum) random number
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
			if (color_jpl_assign_color(color, &colors[i], localmax, localmin)) ++left;
#elif defined(COLOR_MAX_INDEPENDENT_SET)
			if (color_jpl_assign_color(color, &colors[i], localmax)) ++left;
#endif
		}
		grid.sync();
	}
}*/

__device__ bool color_jpl_ingore_neighbor(const int c, const int i, const int j, const int jc) {
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
	return ((jc != -1) && (jc != c) && (jc != c + 1)) || (i == j);
#elif defined(COLOR_MAX_INDEPENDENT_SET)
	return ((jc != -1) && (jc != c)) || (i == j);
#endif
}

__device__ bool color_jpl_assign_color(const int c, int* color_i, const bool localmax, const bool localmin) {
#ifdef COLOR_MIN_MAX_INDEPENDENT_SET
	if (localmin) *color_i = c + 1;
#endif
	if (localmax) *color_i = c;

	return localmax || localmin;
}

int color_cusparse(int const n, const int* Ao, const int* Ac, int* colors, int resetCount) {
	float* dAv;
	int* dAo;
	int* dAc;
	int* dColors;

	int c;
	float fractionToColor = 1.0;

	cusparseStatus_t status;
	cusparseHandle_t handle;
	cusparseMatDescr_t matrixDesc;
	cusparseColorInfo_t colorInfo;
	
	Benchmark& bm = *Benchmark::getInstance(resetCount);
	bm.sampleTime();

	status = cusparseCreate(&handle);
	status = cusparseCreateMatDescr(&matrixDesc);
	status = cusparseCreateColorInfo(&colorInfo);

	size_t occupancyBytes = calc_cusparse_memory_occupancy(0, n, Ao);
	float occupancyPerc = checkDeviceMemOccupancy(occupancyBytes);
	if (occupancyPerc > 1.0) {
		printf("Required %.2f%% of global memory.\n", occupancyPerc * 100);
		std::cout << "ERROR: Unavailable resources. Program will be stopped." << std::endl;
		return -1;
	}

	CUDA_SAFE_CALL(cudaMalloc(&dAo, (n + 1) * sizeof(*dAo)));
	CUDA_SAFE_CALL(cudaMemcpy(dAo, Ao, (n + 1) * sizeof(*Ao), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc(&dAc, Ao[n] * sizeof(*dAc)));
	CUDA_SAFE_CALL(cudaMemcpy(dAc, Ac, Ao[n] * sizeof(*Ac), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc(&dColors, n * sizeof(*dColors)));
	CUDA_SAFE_CALL(cudaMemcpy(dColors, colors, n * sizeof(*colors), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc(&dAv, Ao[n] * sizeof(*dAv)));

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
