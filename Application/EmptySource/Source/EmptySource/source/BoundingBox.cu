#pragma once


#include "..\Source\EmptySource\include\Core.h"
#include "..\Source\EmptySource\include\CoreTypes.h"

#include "..\Source\EmptySource\include\Utility\CUDAUtility.h"
#include "..\Source\EmptySource\include\Utility\Timer.h"
#include "..\Source\EmptySource\include\Math\Math.h"
#include "..\Source\EmptySource\include\Mesh.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

template<class T>
struct SharedMemory {
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

/***********************/
/* BOUNDING BOX STRUCT */
/***********************/
struct BoundingBox {
	float3 LowLeft, UpRight;

	// --- Empty box constructor
	__host__ __device__ 
	BoundingBox() : LowLeft(make_float3(0, 0, 0)), UpRight(make_float3(0, 0, 0)) {}

	// --- Construct a box from a single point
	__host__ __device__
	BoundingBox(const float3 &point) : LowLeft(point), UpRight(point) {}

	// --- Construct a box from a pair of points
	__host__ __device__ 
	BoundingBox(const float3 &ll, const float3 &ur) : LowLeft(ll), UpRight(ur) {}
};

/*********************************/
/* BOUNDING BOX REDUCTION STRUCT */
/*********************************/
// --- Reduce a pair of bounding boxes (a, b) to a bounding box containing a and b
struct MeshReduction /*: public thrust::binary_function<BoundingBox, BoundingBox, BoundingBox> */ {
	__host__ __device__ BoundingBox operator()(BoundingBox a, BoundingBox b) {
		// --- Lower left corner
		float3 LowLeft = make_float3(MIN(a.LowLeft.x, b.LowLeft.x), MIN(a.LowLeft.y, b.LowLeft.y), MIN(a.LowLeft.z, b.LowLeft.z));
		// --- Upper right corner
		float3 UpRight = make_float3(MAX(a.UpRight.x, b.UpRight.x), MAX(a.UpRight.y, b.UpRight.y), MAX(a.UpRight.z, b.UpRight.z));

		return BoundingBox(LowLeft, UpRight);
	}
}; 

__global__ void InitPoints(int N, MeshVertex* Vertices, float* BoundingBoxes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride) {
		BoundingBoxes[i] = Vertices[i].Position.x;
		// BoundingBoxes[i].x = Vertices[i].Position.x;
		// BoundingBoxes[i].y = Vertices[i].Position.y;
		// BoundingBoxes[i].z = Vertices[i].Position.z;
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
SumKernel(T *iData, T *oData, unsigned int n) {
	// --- Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T *sdata = SharedMemory<T>();

	// --- Perform first level of reduction,
	// --- reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T mySum = 0;

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
			mySum += tile32.shfl_down(mySum, offset);
		}
	}

	// --- Write result for this block to global mem
	if (cta.thread_rank() == 0) oData[blockIdx.x] = mySum;
}

/*
 * Compute the number of threads and blocks to use for the given reduction kernel
 * For the kernels >= 3, we set threads / block to the minimum of maxThreads and
 * n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
 * 6, we observe the maximum specified number of blocks, because each thread in
 * that kernel can process a variable number of elements.
*/
void GetNumBlocksAndThreads(int N, int MaxBlocks, int MaxThreads, int &Blocks, int &Threads) {
	
	CUDA::CheckErrors( cudaDeviceSynchronize() );
	
	// --- Get device capability, to avoid block/grid size exceed the upper bound
	cudaDeviceProp DeviceProperties;
	int Device;
	CUDA::CheckErrors( cudaGetDevice(&Device) );
	CUDA::CheckErrors( cudaGetDeviceProperties(&DeviceProperties, Device) );


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
void ExecuteSumKernel(int N, int Threads, int Blocks, T* diData, T* doData) {
	dim3 dimBlock(Threads, 1, 1);
	dim3 dimGrid(Blocks, 1, 1);

	// --- When there is only one warp per block, we need to allocate two warps
	// --- worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (Threads <= 32) ? 2 * Threads * sizeof(T) : Threads * sizeof(T);

	switch (Threads) {
		case 512: SumKernel<T, 512, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case 256: SumKernel<T, 256, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case 128: SumKernel<T, 128, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case  64: SumKernel<T,  64, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case  32: SumKernel<T,  32, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case  16: SumKernel<T,  16, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case   8: SumKernel<T,   8, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case   4: SumKernel<T,   4, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case   2: SumKernel<T,   2, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
		case   1: SumKernel<T,   1, isPow2> <<< dimGrid, dimBlock, smemSize >>> (diData, doData, N); break;
	}

	// --- Wait for GPU to finish before accessing on host
	CUDA::CheckErrors(cudaDeviceSynchronize());
}

void ExecuteInitPoints(int N, int Threads, int Blocks, MeshVertex* Vertices, float* BoundingBoxes) {
	dim3 dimBlock(Threads, 1, 1);
	dim3 dimGrid(Blocks, 1, 1);

	InitPoints <<< dimGrid, dimBlock >>> (N, Vertices, BoundingBoxes);

	// --- Wait for GPU to finish before accessing on host
	CUDA::CheckErrors(cudaDeviceSynchronize());
}

/********/
/* MAIN */
/********/
int FindBoundingBox(int N, MeshVertex * Vertices) {


	CUDA::CheckErrors( cudaProfilerStart() );

	size_t Size = N * sizeof(MeshVertex);

	Debug::Timer Timer;
	Timer.Start();

	// --- Allocate Memory in Device
	MeshVertex* dVertices;
	CUDA::CheckErrors( cudaMalloc(&dVertices, Size) );
	CUDA::CheckErrors( cudaMemcpy(dVertices, Vertices, Size, cudaMemcpyHostToDevice) );
	
	float* diPoints;
	CUDA::CheckErrors( cudaMalloc(&diPoints, N * sizeof(float)) );
	CUDA::CheckErrors( cudaMemset(diPoints, 0, N * sizeof(float)) );

	float* doPoints;
	CUDA::CheckErrors( cudaMalloc(&doPoints, N * sizeof(float)) );
	CUDA::CheckErrors( cudaMemset(doPoints, 0, N * sizeof(float)) );

	Timer.Stop();
	// Debug::Log(
	// 	Debug::LogDebug, L"CUDA Host allocation of %s durantion: %dms",
	// 	Text::FormattedData((double)N * 2 * sizeof(float), 2).c_str(),
	// 	Timer.GetEnlapsed()
	// );

	Timer.Start();

	// --- Run kernel on N elements on the GPU
	int NumBlocks = 0;
	int NumThreads = 0;
	GetNumBlocksAndThreads(N, 64, 256, NumBlocks, NumThreads);

	// --- Run Kernel to initialize as float3 the mesh data
	ExecuteInitPoints(N, NumThreads, NumBlocks, dVertices, diPoints);

	/// TEST
	float* InFloats = new float[N];
	CUDA::CheckErrors( cudaMemcpy(InFloats, diPoints, N * sizeof(float), cudaMemcpyDeviceToHost) );

	float CPUResult = 0;
	for (int i = 0; i < N; i++) {
		CPUResult += InFloats[i];
	}
	
	float RealCPUResult = 0;
	for (int i = 0; i < N; i++) {
		RealCPUResult += Vertices[i].Position.x;
	}
	///

	if (Math::IsPow2(N)) {
		ExecuteSumKernel<float, true>(N, NumThreads, NumBlocks, diPoints, doPoints);
	} else {
		ExecuteSumKernel<float, false>(N, NumThreads, NumBlocks, diPoints, doPoints);
	}

	// --- Copy final sum from device to host
	float* GPUResults = new float[NumBlocks];
	float GPUResult = 0;
	CUDA::CheckErrors( cudaMemcpy(GPUResults, doPoints, NumBlocks * sizeof(float), cudaMemcpyDeviceToHost) );
	for (int i = 0; i < NumBlocks; i++) {
		GPUResult += GPUResults[i];
	}

	// Debug::Log(
	// 	Debug::LogDebug, L"CUDA Device Kernel functions durantion: %dms",
	// 	Timer.GetEnlapsed()
	// );

	CUDA::CheckErrors( cudaProfilerStop() );

	// BoundingBox Initial = BoundingBox(make_float3(Vertices[0].Position.x, Vertices[0].Position.y, Vertices[0].Position.z));

	// --- Compute the bounding box for the point set
	BoundingBox Result = BoundingBox(make_float3(GPUResult, CPUResult, RealCPUResult));

	// --- Free device memory
	CUDA::CheckErrors( cudaFree(dVertices) );
	CUDA::CheckErrors( cudaFree(diPoints) );
	CUDA::CheckErrors( cudaFree(doPoints) );

	Timer.Stop();
	// --- Print output
	Debug::Log(Debug::LogDebug,
		L"Bounding Box in (%ss): LowLeft(%.2f, %.2f, %.2f) UpRight(%.2f, %.2f, %.2f)",
		Text::FormattedUnit(Timer.GetEnlapsedSeconds(), 2).c_str(),
		Result.LowLeft.x, Result.LowLeft.y, Result.LowLeft.z,
		Result.UpRight.x, Result.UpRight.y, Result.UpRight.z
	);

	return 0;
}