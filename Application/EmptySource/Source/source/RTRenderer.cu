
#include "../include/Core.h"
#include "../include/CoreCUDA.h"
#include "../include/CoreTypes.h"

#include "../include/Graphics.h"
#include "../include/Texture2D.h"
#include "../include/Utility/Timer.h"
#include "../include/Math/CoreMath.h"
#include "../include/Mesh.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <curand.h>
#include <curand_kernel.h>
#include <surface_functions.h>
#include <cuda_fp16.h>

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
		if (Stamp < MaxDistance && Stamp > MinDistance) {
			Hit.bHit = true;
			Hit.Stamp = Stamp;
			Hit.Normal = Normal;
			return Hit;
		}

		Stamp = (-b + Discriminant) / a;
		if (Stamp < MaxDistance && Stamp > MinDistance) {
			Hit.bHit = true;
			Hit.Stamp = Stamp;
			Hit.Normal = Normal;
			return Hit;
		}
	}
	
	return Hit;
}

__device__ Vector3 RandomSphericalDir(curandState LocalRandState) {
	Vector3 Direction;
	do {
		Direction = (Vector3(curand_uniform(&LocalRandState), curand_uniform(&LocalRandState), curand_uniform(&LocalRandState)) * 2.F) - Vector3(1, 1, 1);
	} while (Direction.MagnitudeSquared() >= 1.0);
	return Direction;
}

template <unsigned char Bounces>
__device__ Vector4 CastRay(const Ray& RayCast, Vector4 * Spheres, curandState LocalRandState);

template <>
__device__ Vector4 CastRay<0>(const Ray& RayCast, Vector4 * Spheres, curandState LocalRandState);

template <unsigned char Bounces>
__device__ Vector4 CastScreenRay(const Ray& RayCast, Vector4 * Spheres, curandState LocalRandState) {
	RayHit Hit1 = HitSphere(Spheres[0], Spheres[0].w, 0.001F, FLT_MAX, RayCast);
	RayHit Hit2 = HitSphere(Spheres[1], Spheres[1].w, 0.001F, Hit1.Stamp, RayCast);
	RayHit * Hit = (Hit1.bHit && Hit2.bHit) ? (Hit1.Stamp < Hit2.Stamp ? &Hit1 : &Hit2) : (Hit1.bHit ? &Hit1 : &Hit2);

	Vector3 Color = Vector3(0.F);
	if (Hit->bHit) {
		float Diffuse = 1;
		for (int i = 0; i < 1; i++) {
			Vector3 Target = RayCast.PointAt(Hit->Stamp) + Hit->Normal + RandomSphericalDir(LocalRandState) * 0.1F;
			// Target.Normalize();
			Diffuse *= 0.5F;
			// Color = ((Target + 1.F) * 0.5F);
			Color += CastRay<Bounces - 1>(Ray(RayCast.PointAt(Hit->Stamp), Target - RayCast.PointAt(Hit->Stamp)), Spheres, LocalRandState);
		}
		return Vector4(Color * Diffuse, 1);
	}

	return CastRay<0>(RayCast, Spheres, LocalRandState);
}

template <unsigned char Bounces>
__device__ Vector4 CastRay(const Ray& RayCast, Vector4 * Spheres, curandState LocalRandState) {
	RayHit Hit1 = HitSphere(Spheres[0], Spheres[0].w, 0.001F, FLT_MAX, RayCast);
	RayHit Hit2 = HitSphere(Spheres[1], Spheres[1].w, 0.001F, Hit1.Stamp, RayCast);
	RayHit * Hit = (Hit1.bHit && Hit2.bHit) ? (Hit1.Stamp < Hit2.Stamp ? &Hit1 : &Hit2) : ( Hit1.bHit ? &Hit1 : &Hit2 );

	Vector3 Color = Vector3(0.F);
	if (Hit->bHit) {
		Vector3 Target = RayCast.PointAt(Hit->Stamp) + Hit->Normal + RandomSphericalDir(LocalRandState) * 0.1F;
		// Target.Normalize();
		// Color = ((Target + 1.F) * 0.5F);
		Color += CastRay<Bounces - 1>(Ray(RayCast.PointAt(Hit->Stamp), Target - RayCast.PointAt(Hit->Stamp)), Spheres, LocalRandState);
		return Vector4(Color, 1);
	}

	return CastRay<0>(RayCast, Spheres, LocalRandState);
}

template <>
__device__ Vector4 CastRay<0>(const Ray& RayCast, Vector4 * Spheres, curandState LocalRandState) {
	Vector3 NormalizedDirection = RayCast.Direction().Normalized();
	float NHit = 0.5F * (NormalizedDirection.y + 1.F);
	Vector3 Color = Vector3(1.F) * (1.F - NHit) + Vector3(0.5F, 0.7F, 1.F) * NHit * 0.5F;
	return Vector4(Color, 1);
}

__global__ void WirteTextureKernel(
	int2 TextSize, 
	Vector3 LowLeft, 
	Vector3 Horizontal, Vector3 Vertical, 
	Vector3 Origin, 
	curandState * RandState, 
	Vector4 * Spheres) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= TextSize.x || y >= TextSize.y) return;
	int PixelIndex = y * TextSize.x + x;
	curandState LocalRandState = RandState[PixelIndex];

	// int Index = y * TextureDimension.x + x;
	// float4 element = make_float4(0, 0, 0, 0);
	// surf2Dread(&element, SurfaceWrite, x * sizeof(float4), y);
	Vector4 Color = Vector4();
	Vector2 Coord; Ray RayCast;
	const int Samples = 2;
	for (int s = 0; s < Samples; s++) {
		Coord.u = float(x + (curand_uniform(&LocalRandState) - 1) / 2.F) / float(TextSize.x);
		Coord.v = float(y + (curand_uniform(&LocalRandState) - 1) / 2.F) / float(TextSize.y);
		RayCast = Ray(Origin, LowLeft + (Horizontal * Coord.u) + (Vertical * Coord.v));
		Color += CastScreenRay<2>(RayCast, Spheres, LocalRandState);
	}
	Vector4 OutColor = { sqrtf(Color.x / Samples), sqrtf(Color.y / Samples), sqrtf(Color.z / Samples), Color.w / Samples};
	surf2Dwrite(OutColor, SurfaceWrite, x * sizeof(Vector4), y);
}

extern "C"
void LaunchWriteTextureKernel(cudaArray *cudaTextureArray, int2 TextureDim, curandState * RandState, Vector4 * Spheres) {
	dim3 dimBlock(8, 8);
	dim3 dimGrid(TextureDim.x / dimBlock.x + 1, TextureDim.y / dimBlock.y + 1);

	// --- Bind texture array to a writable CUDA surface
	CUDA::Check( cudaBindSurfaceToArray(SurfaceWrite, cudaTextureArray) );

	float WidthRatio = TextureDim.x / 100.F;
	float HeightRatio = TextureDim.y / 100.F;
	WirteTextureKernel <<< dimGrid, dimBlock >>> (
		TextureDim, {-WidthRatio, -HeightRatio, -HeightRatio }, { 2 * WidthRatio, 0, 0}, {0, 2 * HeightRatio, 0}, 0, RandState, Spheres
	);
	CUDA::GetLastCudaError("WriteTextureKernel Failed");
	// --- Wait for GPU to finish
	CUDA::Check( cudaDeviceSynchronize() );
}

int RTRenderToTexture2D(Texture2D * texture, std::vector<Vector4> * Spheres, const void * dRandState) {
	cudaGraphicsResource *cudaTextureResource;
	cudaArray            *cudaTextureArray;
	Vector4              *dSpheres;

	IntVector2 TextureDim = texture->GetDimension();

	CUDA::Check( cudaProfilerStart() );

	// --- Allocate Spheres
	CUDA::Check( cudaMalloc((void **)&dSpheres, Spheres->size() * sizeof(Vector4)) );
	CUDA::Check( cudaMemcpy(dSpheres, &(*Spheres)[0], Spheres->size() * sizeof(Vector4), cudaMemcpyHostToDevice) );

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
		LaunchWriteTextureKernel(cudaTextureArray, { TextureDim.x, TextureDim.y }, (curandState*)dRandState, dSpheres);
	}
	CUDA::Check( cudaGraphicsUnmapResources(1, &cudaTextureResource, 0) );

	// --- Wait for GPU to finish
	CUDA::Check( cudaDeviceSynchronize() );

	Timer.Stop();
	Debug::Log(
		Debug::LogDebug, L"CUDA Texture Write/Read with total volume (%ls): %.2fms",
		Text::FormatUnit(TextureDim.x * TextureDim.y, 3).c_str(),
		Timer.GetEnlapsedMili()
	);

	CUDA::Check( cudaGraphicsUnregisterResource(cudaTextureResource) );

	return 0;
}