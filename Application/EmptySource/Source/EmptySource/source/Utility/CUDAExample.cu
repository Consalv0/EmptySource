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
#include "..\Source\EmptySource\include\Mesh.h"
#include "..\Source\EmptySource\include\BoundingBox.h"

// NVIDIA Reduction Solution https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf 
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

 // Kernel function to add the elements of two arrays
__global__ void Add(int n, MeshVertex *Positions) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    Positions[i].Position.x = Positions[i].Position.y * Positions[i].Position.z;
}

extern "C" bool FindCudaDevice() {
	return CUDA::FindCudaDevice();
}

extern "C" bool RunTest(int N, MeshVertex * Positions) {

	CUDA::CheckCudaErrors(cudaProfilerStart()); 
	
	size_t Size = N * sizeof(MeshVertex);

	Debug::Timer Timer;
	Timer.Start();

	// Allocate Memory in Device
	MeshVertex* dPositions;
	CUDA::CheckCudaErrors( cudaMalloc(&dPositions, Size) );

	CUDA::CheckCudaErrors(cudaMemcpy(dPositions, Positions, Size, cudaMemcpyHostToDevice));

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
	// Init <<< numBlocks, blockSize >>> (N, dPositions);
	// Run kernel on N elements on the GPU
	Add <<< numBlocks, blockSize >>> (N, dPositions);

	// Wait for GPU to finish before accessing on host
	CUDA::CheckCudaErrors( cudaDeviceSynchronize() );

	// Check if kernel execution generated and error
	CUDA::GetLastCudaError("Kernel execution failed");

	CUDA::CheckCudaErrors( cudaMemcpy(Positions, dPositions, Size, cudaMemcpyDeviceToHost) );

	// Free memory
	CUDA::CheckCudaErrors( cudaFree(dPositions) );

	Timer.Stop();
	// Debug::Log(
	// 	Debug::LogDebug, L"CUDA Device Kernel functions durantion: %dms",
	// 	Timer.GetEnlapsed()
	// );

	CUDA::CheckCudaErrors(cudaProfilerStop());

	return 0;
}
