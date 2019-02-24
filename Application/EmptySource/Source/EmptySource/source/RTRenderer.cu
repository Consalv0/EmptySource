
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

#include <curand.h>
#include <curand_kernel.h>
#include <surface_functions.h>

surface<void, cudaSurfaceType2D> SurfaceWrite;

__device__ RayHit HitSphere(const Vector3& Center, const float& Radius, const float& MinDistance, const float& MaxDistance, const Ray& ray) {
	RayHit Hit;
	Vector3 OC = ray.Origin() - Center;
	float a = ray.Direction().Dot(ray.Direction());
	float b = OC.Dot(ray.Direction());
	float c = OC.Dot(OC) - Radius * Radius;
	float Discriminant = sqrtf((b * b) - (a * c));
	
	if (Discriminant >= 0.F) {
		float Stamp = (-b - Discriminant) / a;
		Vector3 Normal = (ray.PointAt(Stamp) - Center) / Radius;
		if (Stamp <= MaxDistance && Stamp >= MinDistance) {
			Hit.bHit = true;
			Hit.Stamp = Stamp;
			Hit.Normal = Normal;
			return Hit;
		}

		Stamp = (-b + Discriminant) / a;
		if (Stamp <= MaxDistance && Stamp >= MinDistance) {
			Hit.bHit = true;
			Hit.Stamp = Stamp;
			Hit.Normal = Normal;
			return Hit;
		}
	}
	
	return Hit;
}

template <unsigned char Bounces>
__device__ Vector4 CastRay(const Ray& ray, Vector4 * Spheres);

template <>
__device__ Vector4 CastRay<0>(const Ray& ray, Vector4 * Spheres);

template <unsigned char Bounces>
__device__ Vector4 CastRay(const Ray& ray, Vector4 * Spheres) {
	RayHit Hit1 = HitSphere(Spheres[0], Spheres[0].w, 0.001F, FLT_MAX, ray);
	RayHit Hit2 = HitSphere(Spheres[1], Spheres[1].w, 0.001F, FLT_MAX, ray);
	RayHit * Hit = (Hit1.bHit && Hit2.bHit) ? (Hit1.Stamp < Hit2.Stamp ? &Hit1 : &Hit2) : ( Hit1.bHit ? &Hit1 : &Hit2 );

	Vector3 Color = CastRay<0>(ray, Spheres);
	if (Hit->bHit) {
		// --- Normal Color ((Hit.Normal + 1.F) * 0.5F)
		Color = (Color + ( CastRay<Bounces - 1>(Ray(ray.PointAt(Hit->Stamp), Vector3::Reflect(ray.Direction(), Hit->Normal)), Spheres) ) ) * 0.5F;
	}

	return Color;
}

template <>
__device__ Vector4 CastRay<0>(const Ray& ray, Vector4 * Spheres) {
	Vector3 NormalizedDirection = ray.Direction().Normalized();
	float NHit = 0.5F * (NormalizedDirection.y + 1.F);
	Vector3 Color = Vector3(1.F) * (1.F - NHit) + Vector3(0.5F, 0.7F, 1.F) * NHit * 0.5F;
	return Color;
}

__global__ void InitRandomKernel(int2 TextureDimension, curandState * RandState) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= TextureDimension.x || y >= TextureDimension.y) return;

	int Index = y * TextureDimension.x + x;
	// --- Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, Index, 0, &RandState[Index]);
}

__global__ void WirteTextureKernel(int2 TextSize, Vector3 LowLeft, Vector3 Horizontal, Vector3 Vertical, Vector3 Origin, Vector4 * Spheres) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= TextSize.x || y >= TextSize.y) return;

	// int Index = y * TextureDimension.x + x;
	// float4 element = make_float4(0, 0, 0, 0);
	// surf2Dread(&element, SurfaceWrite, x * sizeof(float4), y);
	Vector4 Color = Vector4();
	Vector2 Coord; Ray ray;
	// for (int s = 0; s < 4; s++) {
		Coord.u = float(x + 0.25F) / float(TextSize.x);
		Coord.v = float(y + 0.1F) / float(TextSize.y);
		ray = Ray(Origin, LowLeft + (Horizontal * Coord.u) + (Vertical * Coord.v));
		Color += CastRay<2>(ray, Spheres);
		// Coord.u = float(x - 0.5) / float(TextSize.x);
		// Coord.v = float(y - 0.2) / float(TextSize.y);
		// ray = Ray(Origin, LowLeft + (Horizontal * Coord.u) + (Vertical * Coord.v));
		// Color += CastRay<2>(ray, Spheres);
	// }
	surf2Dwrite(Color / 1.0F, SurfaceWrite, x * sizeof(float4), y);
}

extern "C"
void LaunchWriteTextureKernel(cudaArray *cudaTextureArray, int2 TextureDim, curandState * RandState, Vector4 * Spheres) {
	dim3 dimBlock(8, 8);
	dim3 dimGrid(TextureDim.x / dimBlock.x + 1, TextureDim.y / dimBlock.y + 1);

	// --- Bind texture array to a writable CUDA surface
	CUDA::Check( cudaBindSurfaceToArray(SurfaceWrite, cudaTextureArray) );

	// InitRandomKernel <<< dimGrid, dimBlock >>> (TextureDim, RandState);
	// CUDA::GetLastCudaError("InitRandomKernel Failed");
	// // --- Wait for GPU to finish
	// CUDA::Check( cudaDeviceSynchronize() );

	float WidthRatio = TextureDim.x / 100.F;
	float HeightRatio = TextureDim.y / 100.F;
	WirteTextureKernel <<< dimGrid, dimBlock >>> (
		TextureDim, {-WidthRatio, -HeightRatio, -HeightRatio }, { 2 * WidthRatio, 0, 0}, {0, 2 * HeightRatio, 0}, 0, Spheres
	);
	CUDA::GetLastCudaError("WriteTextureKernel Failed");
	// --- Wait for GPU to finish
	CUDA::Check( cudaDeviceSynchronize() );
}


int RayTracingTexture2D(Texture2D * texture, std::vector<Vector4> * Spheres) {
	cudaGraphicsResource *cudaTextureResource;
	cudaArray            *cudaTextureArray;
	curandState          *dRandState;
	Vector4              *dSpheres;

	IntVector2 TextureDim = texture->GetDimension();

	CUDA::Check( cudaProfilerStart() );

	// --- Allocate Spheres
	CUDA::Check( cudaMalloc((void **)&dSpheres, Spheres->size() * sizeof(Vector4)) );
	CUDA::Check( cudaMemcpy(dSpheres, &(*Spheres)[0], Spheres->size() * sizeof(Vector4), cudaMemcpyHostToDevice) );

	// --- Allocate pseudo random values
	CUDA::Check( cudaMalloc((void **)&dRandState, TextureDim.x * TextureDim.y * sizeof(curandState)) );

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
		LaunchWriteTextureKernel(cudaTextureArray, { TextureDim.x, TextureDim.y }, dRandState, dSpheres);
	}
	CUDA::Check( cudaGraphicsUnmapResources(1, &cudaTextureResource, 0) );

	// --- Wait for GPU to finish
	CUDA::Check( cudaDeviceSynchronize() );

	CUDA::Check( cudaFree(dRandState) );

	Timer.Stop();
	Debug::Log(
		Debug::LogDebug, L"CUDA Texture Write/Read with total volume (%s): %dms",
		Text::FormatUnit(TextureDim.x * TextureDim.y, 3).c_str(),
		Timer.GetEnlapsed()
	);

	CUDA::Check( cudaGraphicsUnregisterResource(cudaTextureResource) );

	return 0;
}