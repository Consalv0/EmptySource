////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * Host part of the device code.
 * Compiled with Cuda compiler.
 */

#include "..\Source\EmptySource\include\Core.h"
#include "..\Source\EmptySource\include\CoreTypes.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "..\Source\EmptySource\include\Utility\CUDAUtility.h"
#include "..\Source\EmptySource\include\Utility\Timer.h"
#include "..\Source\EmptySource\include\Math\Math.h"

 // Kernel function to add the elements of two arrays
__global__ void Add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] * y[i];
}

// Kernel function to init the elements of two arrays
__global__ void Init(int n, float *x, float *y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		x[i] = i * 1.0f;
		y[i] = i * 2.0F;
	}
}

extern "C" bool FindCudaDevice() {
	return CUDA::FindCudaDevice();
}

extern "C" bool RunTest(int N, float * x, float * y) {

	CUDA::CheckCudaErrors(cudaProfilerStart());

	Debug::Timer Timer;
	Timer.Start();

	// Allocate Memory in Host
	CUDA::CheckCudaErrors( cudaHostAlloc(&x, N * sizeof(float), cudaHostAllocDefault) );
	CUDA::CheckCudaErrors( cudaHostAlloc(&y, N * sizeof(float), cudaHostAllocDefault) );

	CUDA::CheckCudaErrors( cudaMemset(x, 0, N * sizeof(float)) );
	CUDA::CheckCudaErrors( cudaMemset(y, 0, N * sizeof(float)) );

	Timer.Stop();
	// Debug::Log(
	// 	Debug::LogDebug, L"CUDA Host allocation of %s durantion: %dms",
	// 	Text::FormattedData((double)N * 2 * sizeof(float), 2).c_str(),
	// 	Timer.GetEnlapsed()
	// );

	Timer.Start();
	// Run kernel on N elements on the GPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	// Run Kernel to initialize x and y arrays
	Init <<< numBlocks, blockSize >>> (N, x, y);
	// Run kernel on N elements on the GPU
	Add <<< numBlocks, blockSize >>> (N, x, y);

	// Wait for GPU to finish before accessing on host
	CUDA::CheckCudaErrors(cudaDeviceSynchronize());

	Debug::Log(Debug::LogDebug, L"Test Vector[%d] %s", 1500, Vector2(x[1500], y[1500]).ToString().c_str());

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	Debug::Log(Debug::LogDebug, L"Max error: %.2f", maxError);

	// Check if kernel execution generated and error
	CUDA::GetLastCudaError("Kernel execution failed");

	// Free memory
	CUDA::CheckCudaErrors(cudaFreeHost(x));
	CUDA::CheckCudaErrors(cudaFreeHost(y));

	Timer.Stop();
	// Debug::Log(
	// 	Debug::LogDebug, L"CUDA Device Kernel functions durantion: %dms",
	// 	Timer.GetEnlapsed()
	// );

	CUDA::CheckCudaErrors(cudaProfilerStop());

	return 0;
}
