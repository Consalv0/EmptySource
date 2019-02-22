
#include "..\Source\EmptySource\include\Core.h"
#include "..\Source\EmptySource\include\Graphics.h"
#include "..\Source\EmptySource\include\CoreTypes.h"

#include "..\Source\EmptySource\include\Texture2D.h"
#include "..\Source\EmptySource\include\Utility\CUDAUtility.h"
#include "..\Source\EmptySource\include\Utility\Timer.h"
#include "..\Source\EmptySource\include\Math\Math.h"
#include "..\Source\EmptySource\include\Mesh.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <surface_functions.h>

surface<void, cudaSurfaceType2D> SurfaceWrite;

__device__ Vector4 RayColor(const Ray& r) {
	Vector3 NormalizedDirection = r.Direction() / r.Direction().MagnitudeSquared();
	float t = 0.5F * (NormalizedDirection.y + 1.0F);
	return Vector3(1.0F) * (1.0F - t) + Vector3(0.5F, 0.7F, 1.0F) * t;
}

__global__ void WirteTextureKernel(int2 TextureDimension, Vector3 LowLeft, Vector3 Horizontal, Vector3 Vertical, Vector3 Origin) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= TextureDimension.x || y >= TextureDimension.y) return;

	// float4 element = make_float4(0, 0, 0, 0);
	// surf2Dread(&element, SurfaceWrite, x * sizeof(float4), y);
	float u = float(x) / float(TextureDimension.x);
	float v = float(y) / float(TextureDimension.y);
	Ray ray(Origin, LowLeft + (Horizontal * u) + (Vertical * v));
	surf2Dwrite(RayColor(ray), SurfaceWrite, x * sizeof(float4), y);
}

extern "C"
void LaunchWriteTextureKernel(cudaArray *cudaTextureArray, int2 TextureDim) {
	dim3 dimBlock(8, 8);
	dim3 dimGrid(TextureDim.x / dimBlock.x + 1, TextureDim.y / dimBlock.y + 1);

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

	WirteTextureKernel <<< dimGrid, dimBlock >>> (TextureDim, {-2, -1, -1}, {4, 0, 0}, {0, 2, 0}, 0);

	Error = cudaGetLastError();
	if (Error != cudaSuccess) {
		Debug::Log(
			Debug::LogError, L"%s",
			CharToWChar(cudaGetErrorString(Error))
		);
	}
}


int RayTracingTexture2D(Texture2D* texture) {
	cudaGraphicsResource *cudaTextureResource;
	cudaArray            *cudaTextureArray;

	CUDA::Check( cudaProfilerStart() );

	// --- Register Image (texture) to CUDA Resource
	CUDA::Check( cudaGraphicsGLRegisterImage(&cudaTextureResource,
		texture->GetTextureObject(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore) 
	);

	Debug::Timer Timer;
	Timer.Start();
	// --- Map CUDA resource
	CUDA::Check( cudaGraphicsMapResources(1, &cudaTextureResource, 0) );
	{
		// --- Get mapped array
		CUDA::Check( cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, cudaTextureResource, 0, 0) );
		IntVector2 TextureDim = texture->GetDimension();
		LaunchWriteTextureKernel(cudaTextureArray, { TextureDim.x, TextureDim.y });
	}
	CUDA::Check( cudaGraphicsUnmapResources(1, &cudaTextureResource, 0) );

	// --- Wait for GPU to finish
	CUDA::Check( cudaDeviceSynchronize() );

	Timer.Stop();
	IntVector2 TextureDim = texture->GetDimension();
	Debug::Log(
		Debug::LogDebug, L"CUDA Texture Write/Read with total volume (%s): %dms",
		Text::FormatUnit(TextureDim.x * TextureDim.y, 0).c_str(),
		Timer.GetEnlapsed()
	);

	CUDA::Check( cudaGraphicsUnregisterResource(cudaTextureResource) );

	return 0;
}