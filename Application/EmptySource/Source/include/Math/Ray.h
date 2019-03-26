#pragma once

#include "Vector3.h"

struct RayHit {
public:
	bool bHit;
	float Stamp;
	Vector3 Normal;

	HOST_DEVICE FORCEINLINE RayHit() {
		bHit = false;
#ifndef __CUDACC__
		Stamp = MathConstants::Big_Number;
#else
		Stamp = 3.4e+38f;
#endif
		Normal = Vector3();
	}
};

class Ray {
public:
	Vector3 A;
	Vector3 B;
	
	HOST_DEVICE FORCEINLINE Ray();
	HOST_DEVICE FORCEINLINE Ray(const Vector3& a, const Vector3& b) { A = a; B = b; }
	HOST_DEVICE FORCEINLINE Vector3 Origin() const { return A; }
	HOST_DEVICE FORCEINLINE Vector3 Direction() const { return B; }
	HOST_DEVICE FORCEINLINE Vector3 PointAt(float t) const { return A + (B * t); };
};

#include "Ray.inl"
