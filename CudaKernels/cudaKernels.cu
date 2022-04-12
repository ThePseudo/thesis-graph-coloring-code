#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/count.h>

#include "cudaKernels.h"

#include <algorithm>
#include <iostream>
#include <vector>

#define CUDA_MAX_BLOCKS 2147483647 // Maximum blocks to launch, depending on GPU

__global__ void color_jpl_kernel(int n, int c, const size_t* Ao, const size_t* Ac, const int* randoms, int* colors);

int color_jpl(int const n, const size_t* Ao, const size_t* Ac, int* colors, const int* randoms) {
	cudaError_t err = cudaSuccess;

	size_t* dAo;
	size_t* dAc;
	int* dRandoms;
	int* dColors;
	
	err = cudaMalloc(&dAo, (n+1) * sizeof(*dAo));
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

	err = cudaMemcpy(dAo, Ao, (n+1) * sizeof(*Ao), cudaMemcpyHostToDevice);
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
	
	int c;
	for (c = 0; c < n; ++c) {
		int nt = 256;
		int nb = std::min((n + nt - 1) / nt, CUDA_MAX_BLOCKS);
		color_jpl_kernel<<<nb, nt>>>(n, c, dAo, dAc, dRandoms, dColors);
		err = cudaMemcpy(colors, dColors, n * sizeof(*colors), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			std::cout << "Error9: " << cudaGetErrorString(err) << std::endl;
			goto Error;
		}
		int left = (int)thrust::count(colors, colors + n, -1);
		//std::cout << left << std::endl;
		if (left == 0) break;
	}

Error:
	cudaFree(dAo);
	cudaFree(dAc);
	cudaFree(dRandoms);
	cudaFree(dColors);

	return err == cudaSuccess ? c : -1;
}

__global__ void color_jpl_kernel(int n, int c, const size_t* Ao, const size_t* Ac, const int* randoms, int* colors) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x;
		i < n;
		i += blockDim.x * gridDim.x)
	{
		bool f = true; // true if you have max random

		// ignore nodes colored earlier
		if (colors[i] != -1) continue;

		int ir = randoms[i];

		// look at neighbors to check their random number
		for (int k = Ao[i]; k < Ao[i + 1]; k++) {
			// ignore nodes colored earlier (and yourself)
			int j = Ac[k];
			int jc = colors[j];
			if (((jc != -1) && (jc != c)) || (i == j)) continue;
			int jr = randoms[j];
			if (ir <= jr) f = false;
		}

		// assign color if you have the maximum random number
		if (f) colors[i] = c;
	}
}