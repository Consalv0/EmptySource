
#include "..\Source\EmptySource\include\Core.h"
#include "..\Source\EmptySource\include\CoreCUDA.h"
#include "..\Source\EmptySource\include\CoreTypes.h"

#include "..\Source\EmptySource\include\Graphics.h"
#include "..\Source\EmptySource\include\Utility\Timer.h"
#include "..\Source\EmptySource\include\Math\CoreMath.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <curand.h>
#include <curand_kernel.h>

__global__ void InitRandomKernel(int2 Dimension, curandState * RandState) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= Dimension.x || y >= Dimension.y) return;

	int Index = y * Dimension.x + x;
	// --- Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, Index, 0, &RandState[Index]);
}

extern "C"
void LaunchRandomKernel(int2 Dimension, curandState * RandState) {
	dim3 dimBlock(8, 8);
	dim3 dimGrid(Dimension.x / dimBlock.x + 1, Dimension.y / dimBlock.y + 1);

	InitRandomKernel <<< dimGrid, dimBlock >>> (Dimension, RandState);
	CUDA::GetLastCudaError("InitRandomKernel Failed");
	
	// --- Wait for GPU to finish
	CUDA::Check( cudaDeviceSynchronize() );
}

const void * GetRandomArray(IntVector2 Dimension) {
	curandState          *dRandState;

	CUDA::Check(cudaProfilerStart());

	// --- Allocate pseudo random values
	CUDA::Check(cudaMalloc((void **)&dRandState, Dimension.x * Dimension.y * sizeof(curandState)));

	Debug::Timer Timer;
	Timer.Start();
	LaunchRandomKernel({ Dimension.x, Dimension.y }, dRandState);
	
	// --- Wait for GPU to finish
	CUDA::Check(cudaDeviceSynchronize());

	Timer.Stop();
	Debug::Log(
		Debug::LogDebug, L"CUDA Random values with total volume (%s): %dms",
		Text::FormatUnit(Dimension.x * Dimension.y, 3).c_str(),
		Timer.GetEnlapsedMili()
	);

	return dRandState;
}