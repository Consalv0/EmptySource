
#include "..\Source\EmptySource\include\Core.h"
#include "..\Source\EmptySource\include\Graphics.h"
#include "..\Source\EmptySource\include\CoreTypes.h"

#include "..\Source\EmptySource\include\Utility\CUDAUtility.h"
#include "..\Source\EmptySource\include\Utility\Timer.h"
#include "..\Source\EmptySource\include\Math\Math.h"
#include "..\Source\EmptySource\include\Texture3D.h"
#include "..\Source\EmptySource\include\Mesh.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <surface_functions.h>

surface<void, cudaSurfaceType3D> SurfaceWrite; 

__global__ void WirteKernel(int N, MeshVertex *Vertices, dim3 TextureDimension) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < N) {
		uint3 Position = make_uint3(Vertices[index].Position.x, Vertices[index].Position.y, Vertices[index].Position.z);
		if (Position.x >= TextureDimension.x || Position.y >= TextureDimension.y || Position.z >= TextureDimension.z) {
			return;
		}

		float4 element = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
		surf3Dread(&element, SurfaceWrite, Position.x * sizeof(float4), Position.y, Position.z);
		element = make_float4(
			Vertices[index].Normal.x * 0.5F + element.x * 0.5F,
			Vertices[index].Normal.y * 0.5F + element.y * 0.5F,
			Vertices[index].Normal.z * 0.5F + element.z * 0.5F,
			0
		);
		surf3Dwrite(element, SurfaceWrite, Position.x * sizeof(float4), Position.y, Position.z);
	}
}

extern "C"
void LaunchWriteKernel(int N, MeshVertex * dVertices, cudaArray *cudaTextureArray, dim3 TextureDim) {
	dim3 dimBlock(8, 8, 8);
	dim3 dimGrid(TextureDim.x / dimBlock.x, TextureDim.y / dimBlock.y, TextureDim.z / dimBlock.z);

	cudaError_t Error;

	// --- Bind voxel array to a writable CUDA surface
	Error = cudaBindSurfaceToArray(SurfaceWrite, cudaTextureArray);
	if (Error != cudaSuccess) {
		Debug::Log(
			Debug::LogError, L"%s",
			CharToWChar(cudaGetErrorString(Error))
		);
		return;
	}

	WirteKernel<<< dimGrid, dimBlock >>> (N, dVertices, TextureDim);

	Error = cudaGetLastError();
	if (Error != cudaSuccess) {
		Debug::Log(
			Debug::LogError, L"%s",
			CharToWChar(cudaGetErrorString(Error))
		);
	}
}


int VoxelizeToTexture3D(Texture3D* texture, int N, MeshVertex * Vertices) {
	cudaGraphicsResource *cudaTextureResource;
	cudaArray            *cudaTextureArray;
	MeshVertex           *dVertices;

	size_t Size = N * sizeof(MeshVertex);

	CUDA::Check( cudaProfilerStart() );

	Debug::Timer Timer;
	Timer.Start();

	// --- Allocate Memory in Device
	CUDA::Check( cudaMalloc(&dVertices, Size) );
	CUDA::Check( cudaMemcpy(dVertices, Vertices, Size, cudaMemcpyHostToDevice) );

	// --- Register Image (texture) to CUDA Resource
	CUDA::Check( cudaGraphicsGLRegisterImage(&cudaTextureResource,
		texture->GetTextureObject(), GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore) 
	);

	Timer.Stop();
	Debug::Log(
		Debug::LogDebug, L"CUDA Host allocation of %s durantion: %dms",
		Text::FormatData((double)N * 2 * sizeof(MeshVertex) + Size, 2).c_str(),
		Timer.GetEnlapsed()
	);

	Timer.Start();
	// --- Map CUDA resource
	CUDA::Check( cudaGraphicsMapResources(1, &cudaTextureResource, 0) );
	{
		// --- Get mapped array
		CUDA::Check( cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, cudaTextureResource, 0, 0) );
		IntVector3 TextureDim = texture->GetDimension();
		LaunchWriteKernel(N, dVertices, cudaTextureArray, dim3(TextureDim.x, TextureDim.y, TextureDim.z));
	}
	CUDA::Check( cudaGraphicsUnmapResources(1, &cudaTextureResource, 0) );

	// --- Wait for GPU to finish
	CUDA::Check( cudaDeviceSynchronize() );

	Timer.Stop();
	IntVector3 TextureDim = texture->GetDimension();
	Debug::Log(
		Debug::LogDebug, L"CUDA Texture Write with total volume (%s): %dms",
		Text::FormatUnit(TextureDim.x * TextureDim.y * TextureDim.z, 0).c_str(),
		Timer.GetEnlapsed()
	);

	CUDA::Check( cudaGraphicsUnregisterResource(cudaTextureResource) );

	return 0;
}