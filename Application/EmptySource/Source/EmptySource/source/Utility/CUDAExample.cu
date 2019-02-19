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
#include "..\Source\EmptySource\include\Graphics.h"
#include "..\Source\EmptySource\include\CoreTypes.h"

#include "..\Source\EmptySource\include\Utility\CUDAUtility.h"
#include "..\Source\EmptySource\include\Utility\Timer.h"
#include "..\Source\EmptySource\include\Math\Math.h"
#include "..\Source\EmptySource\include\Mesh.h"

 // --- Kernel function to add the elements of two arrays
__global__ void Add(int elementCount, MeshVertex *Positions) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < elementCount; i += stride) {
	  Positions[i].Position.x = Positions[i].Position.y * Positions[i].Position.z;
  }
}

bool RunTest(int N, MeshVertex * Positions) {

	CUDA::Check(cudaProfilerStart()); 
	
	size_t Size = N * sizeof(MeshVertex);

	Debug::Timer Timer;
	Timer.Start();

	// Allocate Memory in Device
	MeshVertex* dPositions;
	CUDA::Check( cudaMalloc(&dPositions, Size) );
	CUDA::Check( cudaMemcpy(dPositions, Positions, Size, cudaMemcpyHostToDevice) );

	Timer.Stop();
	// Debug::Log(
	// 	Debug::LogDebug, L"CUDA Host allocation of %s durantion: %dms",
	// 	Text::FormatData((double)N * 2 * sizeof(float), 2).c_str(),
	// 	Timer.GetEnlapsed()
	// );

	Timer.Start();
	// Run kernel on N elements on the GPU
	const unsigned int blockSize = 256;
	const unsigned int numBlocks = (N + blockSize - 1) / blockSize;

	// Run Kernel to initialize x and y arrays
	// Init <<< numBlocks, blockSize >>> (N, dPositions);
	// Run kernel on N elements on the GPU
	Add<<< numBlocks, blockSize >>> (N, dPositions);

	// Wait for GPU to finish before accessing on host
	CUDA::Check( cudaDeviceSynchronize() );

	// Check if kernel execution generated and error
	CUDA::GetLastCudaError("Kernel execution failed");

	CUDA::Check( cudaMemcpy(Positions, dPositions, Size, cudaMemcpyDeviceToHost) );

	// Free memory
	CUDA::Check( cudaFree(dPositions) );

	Timer.Stop();
	// Debug::Log(
	// 	Debug::LogDebug, L"CUDA Device Kernel functions durantion: %dms",
	// 	Timer.GetEnlapsed()
	// );

	CUDA::Check(cudaProfilerStop());

	return 0;
}
