﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/count.h>

#include "cudaKernels.h"
#include "cusparse.h"

#include <algorithm>
#include <iostream>
#include <vector>

#define CUDA_MAX_BLOCKS 2147483647 // Maximum blocks to launch, depending on GPU

__global__ void create_independent_set_kernel(int n, const int* Ao, const int* Ac, const int* randoms, const int* colors, unsigned int* set);
__global__ void expand_to_maximal_independent_set_kernel(int n, const int* Ao, const int* Ac, const int* colors, unsigned int* set);
__global__ void color_jpl_kernel(int n, int c, int* colors, const unsigned int* set);

int color_jpl(int const n, const int* Ao, const int* Ac, int* colors, const int* randoms) {
	cudaError_t err = cudaSuccess;

	int* dAo;
	int* dAc;
	int* dRandoms;
	int* dColors;
	unsigned int* dSet;
	Benchmark& bm = *Benchmark::getInstance();

	err = cudaMalloc(&dAo, (n + 1) * sizeof(*dAo));
	if (err != cudaSuccess) {
		std::cout << "Error1: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMalloc(&dAc, Ao[n] * sizeof(*dAc));
	if (err != cudaSuccess) {
		std::cout << "Error2: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMalloc(&dRandoms, n * sizeof(*dRandoms));
	if (err != cudaSuccess) {
		std::cout << "Error3: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMalloc(&dColors, n * sizeof(*dColors));
	if (err != cudaSuccess) {
		std::cout << "Error4: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMalloc(&dSet, n * sizeof(*dSet));
	if (err != cudaSuccess) {
		std::cout << "Error4: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}

	err = cudaMemcpy(dAo, Ao, (n + 1) * sizeof(*Ao), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Error5: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMemcpy(dAc, Ac, Ao[n] * sizeof(*Ac), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Error6: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMemcpy(dRandoms, randoms, n * sizeof(*randoms), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Error7: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMemcpy(dColors, colors, n * sizeof(*colors), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Error8: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMemset(dSet, 0x0, n * sizeof(*dSet));
	if (err != cudaSuccess) {
		std::cout << "Error8: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}

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

		err = cudaMemcpy(colors, dColors, n * sizeof(*colors), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cout << "Error9: " << cudaGetErrorString(err) << std::endl;
			goto Error;
		}
		bm.sampleTimeToFlag(3);

		left = (int)thrust::count(colors, colors + n, -1);
	}

Error:
	cudaFree(dAo);
	cudaFree(dAc);
	cudaFree(dRandoms);
	cudaFree(dColors);
	cudaFree(dSet);

	return err == cudaSuccess ? c : -1;
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
	cudaError_t err = cudaSuccess;

	float* dAv;
	int* dAo;
	int* dAc;
	int* dColors;

	Benchmark& bm = *Benchmark::getInstance();
	bm.sampleTime();

	err = cudaMalloc(&dAo, (n + 1) * sizeof(*dAo));
	if (err != cudaSuccess) {
		std::cout << "Error1: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMemcpy(dAo, Ao, (n + 1) * sizeof(*Ao), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Error5: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}

	err = cudaMalloc(&dAc, Ao[n] * sizeof(*dAc));
	if (err != cudaSuccess) {
		std::cout << "Error2: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMemcpy(dAc, Ac, Ao[n] * sizeof(*Ac), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Error6: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}

	err = cudaMalloc(&dColors, n * sizeof(*dColors));
	if (err != cudaSuccess) {
		std::cout << "Error4: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}
	err = cudaMemcpy(dColors, colors, n * sizeof(*colors), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cout << "Error8: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}

	err = cudaMalloc(&dAv, Ao[n] * sizeof(*dAv));
	if (err != cudaSuccess) {
		std::cout << "Error1: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}

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

	err = cudaMemcpy(colors, dColors, n * sizeof(*colors), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "Error9: " << cudaGetErrorString(err) << std::endl;
		goto Error;
	}

Error:
	bm.sampleTimeToFlag(3);
	cudaFree(dAv);
	cudaFree(dAo);
	cudaFree(dAc);
	cudaFree(dColors);

	cusparseDestroyMatDescr(matrixDesc);
	cusparseDestroyColorInfo(colorInfo);
	cusparseDestroy(handle);

	return err == cudaSuccess ? c : -1;
}