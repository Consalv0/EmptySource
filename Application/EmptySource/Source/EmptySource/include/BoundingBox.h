#pragma once

#include "..\include\Math\Math.h"

struct BoundingBox {
// #ifdef __CUDACC__
// 	float3 LowerLeft, UpperRight;
// 
// 	__host__ __device__ BoundinBox() {} ;
// 
// 	//* Construct a box from a pair of points
// 	__host__ __device__ BoundingBox(const float3 &LowLft, const float3 &UppRgth) : LowerLeft(LowLft), UpperRight(UppRgth) {} ;
// 
// #else
	Vector3 LowerLeft, UpperRight;

	BoundingBox() : LowerLeft(), UpperRight() {};

	//* Construct a box from a pair of points
	BoundingBox(const Vector3 &LowLft, const Vector3 &UppRgth) : LowerLeft(LowLft), UpperRight(UppRgth) {} ;
// #endif // __CUDACC__
};