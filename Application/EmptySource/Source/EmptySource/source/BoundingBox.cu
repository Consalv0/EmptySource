#pragma once


#include "..\Source\EmptySource\include\Core.h"
#include "..\Source\EmptySource\include\CoreCUDA.h"
#include "..\Source\EmptySource\include\CoreTypes.h"

#include "..\Source\EmptySource\include\Graphics.h"
#include "..\Source\EmptySource\include\Utility\Timer.h"
#include "..\Source\EmptySource\include\Math\CoreMath.h"
#include "..\Source\EmptySource\include\Mesh.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MIN(A, B) ( (( A ) < ( B )) ? (A) : (B) )
#define MAX(A, B) ( (( A ) > ( B )) ? (A) : (B) )

template<class T>
struct SharedMemory {
	__device__ inline operator T *() {
		extern __shared__ T __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ T __smem[];
		return (T *)__smem;
	}
};

/***********************/
/* BOUNDING BOX STRUCT */
/***********************/
struct BoundingBox {
	float3 LowLeft, UpRight;

	// --- Empty box constructor
	// __host__ __device__ 
	// BoundingBox() : LowLeft(make_float3(0, 0, 0)), UpRight(make_float3(0, 0, 0)) {}

	// --- Construct a box from a single point
	__host__ __device__
	BoundingBox(const float3 &point) : LowLeft(point), UpRight(point) {}

	// --- Construct a box from a pair of points
	__host__ __device__ 
	BoundingBox(const float3 &ll, const float3 &ur) : LowLeft(ll), UpRight(ur) {}

	__host__ __device__
	inline BoundingBox& operator+=(const BoundingBox & Other) {
		LowLeft.x += Other.LowLeft.x;
		LowLeft.y += Other.LowLeft.y;
		LowLeft.z += Other.LowLeft.z;

		UpRight.x += Other.UpRight.x;
		UpRight.y += Other.UpRight.y;
		UpRight.z += Other.UpRight.z;

		return *this;
	}

	__host__ __device__
	inline BoundingBox operator+(const BoundingBox & Other) const {
		BoundingBox Result = BoundingBox(Other);
		Result.LowLeft.x = LowLeft.x + Other.LowLeft.x;
		Result.LowLeft.y = LowLeft.y + Other.LowLeft.y;
		Result.LowLeft.z = LowLeft.z + Other.LowLeft.z;

		Result.UpRight.x = UpRight.x + Other.UpRight.x;
		Result.UpRight.y = UpRight.y + Other.UpRight.y;
		Result.UpRight.z = UpRight.z + Other.UpRight.z;

		return Result;
	}

	__host__ __device__
	inline BoundingBox operator<(const BoundingBox & Other) const {
		BoundingBox Result = BoundingBox(Other);
		Result.LowLeft.x = MIN(LowLeft.x, Other.LowLeft.x);
		Result.LowLeft.y = MIN(LowLeft.y, Other.LowLeft.y);
		Result.LowLeft.z = MIN(LowLeft.z, Other.LowLeft.z);

		Result.UpRight.x = MAX(UpRight.x, Other.UpRight.x);
		Result.UpRight.y = MAX(UpRight.y, Other.UpRight.y);
		Result.UpRight.z = MAX(UpRight.z, Other.UpRight.z);

		return Result;
	}

	__host__ __device__
	inline BoundingBox& operator<=(const BoundingBox & Other) {
		LowLeft.x = MIN(LowLeft.x, Other.LowLeft.x);
		LowLeft.y = MIN(LowLeft.y, Other.LowLeft.y);
		LowLeft.z = MIN(LowLeft.z, Other.LowLeft.z);
		
		UpRight.x = MAX(UpRight.x, Other.UpRight.x);
		UpRight.y = MAX(UpRight.y, Other.UpRight.y);
		UpRight.z = MAX(UpRight.z, Other.UpRight.z);

		return *this;
	}
};

__global__ void InitBoundingBoxes(int N, MeshVertex* Vertices, BoundingBox* BoundingBoxes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride) {
		BoundingBoxes[i].LowLeft.x = Vertices[i].Position.x;
		BoundingBoxes[i].LowLeft.y = Vertices[i].Position.y;
		BoundingBoxes[i].LowLeft.z = Vertices[i].Position.z;
		BoundingBoxes[i].UpRight = BoundingBoxes[i].LowLeft;
	}
}

/*
 * NVIDIA Reduction Solution https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf 
 * 
 * This version adds multiple elements per thread sequentially.  This reduces the overall
 * cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
 * (Brent's Theorem optimization)
 * 
 * Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
 * In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
 * If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
SumKernel(T *iData, T *oData, T Init, unsigned int n) {
	// --- Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T *sdata = SharedMemory<T>();

	// --- Perform first level of reduction,
	// --- reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T mySum = Init;

	// --- We reduce multiple elements per thread.  The number is determined by the
	// --- number of active thread blocks (via gridDim).  More blocks will result
	// --- in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
		mySum += iData[i];

		// --- Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			mySum += iData[i + blockSize];

		i += gridSize;
	}

	// --- Each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	cg::sync(cta);


	// --- Do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256)) {
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128)) {
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64)) {
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32) {
		// --- Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64) mySum += sdata[tid + 32];
		// --- Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
			mySum.LowLeft.x += tile32.shfl_down(mySum.LowLeft.x, offset);
			mySum.LowLeft.y += tile32.shfl_down(mySum.LowLeft.y, offset);
			mySum.LowLeft.z += tile32.shfl_down(mySum.LowLeft.z, offset);

			mySum.UpRight.x += tile32.shfl_down(mySum.UpRight.x, offset);
			mySum.UpRight.y += tile32.shfl_down(mySum.UpRight.y, offset);
			mySum.UpRight.z += tile32.shfl_down(mySum.UpRight.z, offset);
		}
	}

	// --- Write result for this block to global mem
	if (cta.thread_rank() == 0) oData[blockIdx.x] = mySum;
}

/*
 * NVIDIA Reduction Solution https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf 
 * 
 * This version adds multiple elements per thread sequentially.  This reduces the overall
 * cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
 * (Brent's Theorem optimization)
 * 
 * Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
 * In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
 * If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
MinKernel(T *iData, T *oData, T Init, unsigned int n) {
	// --- Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T *sdata = SharedMemory<T>();

	// --- Perform first level of reduction,
	// --- reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T myMin = Init;

	// --- We reduce multiple elements per thread.  The number is determined by the
	// --- number of active thread blocks (via gridDim).  More blocks will result
	// --- in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
		myMin <= iData[i];

		// --- Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			myMin <= iData[i + blockSize];

		i += gridSize;
	}

	// --- Each thread puts its local sum into shared memory
	sdata[tid] = myMin;
	cg::sync(cta);


	// --- Do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256)) {
		sdata[tid] = myMin = (myMin < sdata[tid + 256]);
	}

	cg::sync(cta);

	if ((blockSize >= 256) && (tid < 128)) {
		sdata[tid] = myMin = (myMin < sdata[tid + 128]);
	}

	cg::sync(cta);

	if ((blockSize >= 128) && (tid < 64)) {
		sdata[tid] = myMin = (myMin < sdata[tid + 64]);
	}

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	if (cta.thread_rank() < 32) {
		// --- Fetch final intermediate min from 2nd warp
		if (blockSize >= 64) myMin <= sdata[tid + 32];

		// --- Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0 && (i + offset) < n; offset /= 2) {
			myMin.LowLeft.x = MIN(myMin.LowLeft.x, tile32.shfl_down(myMin.LowLeft.x, offset));
			myMin.LowLeft.y = MIN(myMin.LowLeft.y, tile32.shfl_down(myMin.LowLeft.y, offset));
			myMin.LowLeft.z = MIN(myMin.LowLeft.z, tile32.shfl_down(myMin.LowLeft.z, offset));
			
			myMin.UpRight.x = MAX(myMin.UpRight.x, tile32.shfl_down(myMin.UpRight.x, offset));
			myMin.UpRight.y = MAX(myMin.UpRight.y, tile32.shfl_down(myMin.UpRight.y, offset));
			myMin.UpRight.z = MAX(myMin.UpRight.z, tile32.shfl_down(myMin.UpRight.z, offset));
		}
	}

	// --- Write result for this block to global mem
	if (cta.thread_rank() == 0) oData[blockIdx.x] = myMin;
}

/*
 * Compute the number of threads and blocks to use for the given reduction kernel
 * For the kernels >= 3, we set threads / block to the minimum of maxThreads and
 * n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
 * 6, we observe the maximum specified number of blocks, because each thread in
 * that kernel can process a variable number of elements.
*/
void GetNumBlocksAndThreads(int N, int MaxBlocks, int MaxThreads, int &Blocks, int &Threads) {
	
	CUDA::Check( cudaDeviceSynchronize() );
	
	// --- Get device capability, to avoid block/grid size exceed the upper bound
	cudaDeviceProp DeviceProperties;
	int Device;
	CUDA::Check( cudaGetDevice(&Device) );
	CUDA::Check( cudaGetDeviceProperties(&DeviceProperties, Device) );


	Threads = (N < MaxThreads * 2) ? Math::NextPow2((N + 1) / 2) : MaxThreads;
	Blocks = (N + (Threads * 2 - 1)) / (Threads * 2);

	if (((float)Threads * Blocks) > ((float)DeviceProperties.maxGridSize[0] * DeviceProperties.maxThreadsPerBlock)) {
		Debug::Log(Debug::LogWarning, L"Size is too large to perform kernel reduction, please reduce the number!");
	}

	if (Blocks > DeviceProperties.maxGridSize[0]) {
		Debug::Log(
			Debug::LogWarning, L"Grid size (%d) exceeds the device capability (%d), set block size as %d (original %d)",
			Blocks, DeviceProperties.maxGridSize[0], Threads * 2, Threads
		);

		Blocks /= 2;
		Threads *= 2;
	}

	Blocks = Math::Min(MaxBlocks, Blocks);
}

template<class T, bool isPow2>
void ExecuteSumKernel(int N, int Threads, int Blocks, T* diData, T* doData, T Init) {
	dim3 dimBlock(Threads, 1, 1);
	dim3 dimGrid(Blocks, 1, 1);

	// --- When there is only one warp per block, we need to allocate two warps
	// --- worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (Threads <= 32) ? 2 * Threads * sizeof(T) : Threads * sizeof(T);

	switch (Threads) {
		case 512: SumKernel<T, 512, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case 256: SumKernel<T, 256, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case 128: SumKernel<T, 128, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case  64: SumKernel<T,  64, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case  32: SumKernel<T,  32, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case  16: SumKernel<T,  16, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case   8: SumKernel<T,   8, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case   4: SumKernel<T,   4, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case   2: SumKernel<T,   2, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
		case   1: SumKernel<T,   1, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	}

	// --- Wait for GPU to finish
	CUDA::Check(cudaDeviceSynchronize());
}

template<class T, bool isPow2>
void ExecuteMinKernel(int N, int Threads, int Blocks, T* diData, T* doData, T Init) {
	dim3 dimBlock(Threads, 1, 1);
	dim3 dimGrid(Blocks, 1, 1);

	// --- When there is only one warp per block, we need to allocate two warps
	// --- worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (Threads <= 32) ? 2 * Threads * sizeof(T) : Threads * sizeof(T);

	switch (Threads) {
	case 512: MinKernel<T, 512, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case 256: MinKernel<T, 256, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case 128: MinKernel<T, 128, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case  64: MinKernel<T,  64, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case  32: MinKernel<T,  32, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case  16: MinKernel<T,  16, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case   8: MinKernel<T,   8, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case   4: MinKernel<T,   4, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case   2: MinKernel<T,   2, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	case   1: MinKernel<T,   1, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, Init, N); break;
	}

	// --- Wait for GPU to finish
	CUDA::Check(cudaDeviceSynchronize());
}

void ExecuteInitBoundingBoxes(int N, int Threads, int Blocks, MeshVertex* Vertices, BoundingBox* BoundingBoxes) {
	dim3 dimBlock(Threads, 1, 1);
	dim3 dimGrid(Blocks, 1, 1);

	InitBoundingBoxes <<< dimGrid, dimBlock >>> (N, Vertices, BoundingBoxes);

	// --- Wait for GPU to finish
	CUDA::Check(cudaDeviceSynchronize());
}

// --- MAIN
int FindBoundingBox(int N, MeshVertex * Vertices) {


	CUDA::Check( cudaProfilerStart() );

	size_t Size = N * sizeof(MeshVertex);

	Debug::Timer Timer;
	Timer.Start();

	// --- Allocate Memory in Device
	MeshVertex* dVertices;
	CUDA::Check( cudaMalloc(&dVertices, Size) );
	CUDA::Check( cudaMemcpy(dVertices, Vertices, Size, cudaMemcpyHostToDevice) );
	
	BoundingBox* diBBox;
	CUDA::Check( cudaMalloc(&diBBox, N * sizeof(BoundingBox)) );

	BoundingBox* doBBox;
	CUDA::Check( cudaMalloc(&doBBox, N * sizeof(BoundingBox)) );

	Timer.Stop();
	Debug::Log(
		Debug::LogDebug, L"CUDA Host allocation of %s durantion: %dms",
		Text::FormatData((double)N * 2 * sizeof(BoundingBox) + Size, 2).c_str(),
		Timer.GetEnlapsed()
	);

	Timer.Start();

	// --- Run kernel on N elements on the GPU
	int NumBlocks = 0;
	int NumThreads = 0;
	GetNumBlocksAndThreads(N, 64, 256, NumBlocks, NumThreads);

	// --- Run Kernel to initialize as float3 the mesh data
	ExecuteInitBoundingBoxes(N, NumThreads, NumBlocks, dVertices, diBBox);

	//
	BoundingBox Initial = BoundingBox(make_float3(Vertices[0].Position.x, Vertices[0].Position.y, Vertices[0].Position.z));

	/// TEST
	// BoundingBox* InBoundBox = (BoundingBox*) malloc(N * sizeof(BoundingBox));
	// CUDA::Check( cudaMemcpy(InBoundBox, diBBox, N * sizeof(BoundingBox), cudaMemcpyDeviceToHost) );

	// BoundingBox CPUResult = Initial;
	// for (int i = 0; i < N; i++)
	// 	CPUResult <= BoundingBox(make_float3(Vertices[i].Position.x, Vertices[i].Position.y, Vertices[i].Position.z));
	// 
	// Debug::Log(Debug::LogDebug,
	// 	L"Bounding Box CPU Result: LowLeft(%.4f, %.4f, %.4f) UpRight(%.4f, %.4f, %.4f)",
	// 	CPUResult.LowLeft.x, CPUResult.LowLeft.y, CPUResult.LowLeft.z,
	// 	CPUResult.UpRight.x, CPUResult.UpRight.y, CPUResult.UpRight.z
	// );

	// BoundingBox GPUCPUResult = Initial;
	// for (int i = 0; i < N; i++)
	// 	GPUCPUResult <= InBoundBox[i]; 

	// Debug::Log(Debug::LogDebug,
	// 	L"Bounding Box GPUCPU Result: LowLeft(%.4f, %.4f, %.4f) UpRight(%.4f, %.4f, %.4f)",
	// 	GPUCPUResult.LowLeft.x, GPUCPUResult.LowLeft.y, GPUCPUResult.LowLeft.z,
	// 	GPUCPUResult.UpRight.x, GPUCPUResult.UpRight.y, GPUCPUResult.UpRight.z
	// );
	///
	
	if (Math::IsPow2(N)) {
		ExecuteMinKernel<BoundingBox, true>(N, NumThreads, NumBlocks, diBBox, doBBox, Initial);
	} else {
		ExecuteMinKernel<BoundingBox, false>(N, NumThreads, NumBlocks, diBBox, doBBox, Initial);
	}

	// --- Copy final result from device to host
	std::vector<BoundingBox> GPUResults = std::vector<BoundingBox>(NumBlocks, Initial);
	CUDA::Check( cudaMemcpy(&GPUResults[0], doBBox, NumBlocks * sizeof(BoundingBox), cudaMemcpyDeviceToHost) );
	
	BoundingBox GPUResult = GPUResults[0];
	for (int i = 0; i < NumBlocks; i++) { GPUResult <= GPUResults[i]; }

	Timer.Stop();
	Debug::Log(
		Debug::LogDebug, L"CUDA Device Kernel functions durantion: %dms",
		Timer.GetEnlapsed()
	);

	CUDA::Check( cudaProfilerStop() );

	// --- Print output
	// Debug::Log(Debug::LogDebug,
	// 	L"Bounding Box in (%ss): LowLeft(%d, %d, %d) UpRight(%d, %d, %d)",
	// 	Text::FormatUnit(Timer.GetEnlapsedSeconds(), 2).c_str(),
	// 	GPUResult.LowLeft.x, GPUResult.LowLeft.y, GPUResult.LowLeft.z,
	// 	GPUResult.UpRight.x, GPUResult.UpRight.y, GPUResult.UpRight.z
	// );
	// Debug::Log(Debug::LogDebug,
	// 	L"Bounding Box Error: LowLeft(%.4f, %.4f, %.4f) UpRight(%.4f, %.4f, %.4f)",
	// 	Text::FormatUnit(Timer.GetEnlapsedSeconds(), 2).c_str(),
	// 	abs(abs(GPUResult.LowLeft.x) - abs(CPUResult.LowLeft.x)), abs(abs(GPUResult.LowLeft.y) - abs(CPUResult.LowLeft.y)),
	// 	abs(abs(GPUResult.LowLeft.z) - abs(CPUResult.LowLeft.z)),
	// 	abs(abs(GPUResult.UpRight.x) - abs(CPUResult.UpRight.x)), abs(abs(GPUResult.UpRight.y) - abs(CPUResult.UpRight.y)),
	// 	abs(abs(GPUResult.UpRight.z) - abs(CPUResult.UpRight.z))
	// );

	// --- Free device memory
	CUDA::Check(cudaFree(dVertices));
	CUDA::Check(cudaFree(diBBox));
	CUDA::Check(cudaFree(doBBox));

	return 0;
}