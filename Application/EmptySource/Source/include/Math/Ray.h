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
		Stamp = MathConstants::BigNumber;
#else
		Stamp = 3.4e+38f;
#endif
		Normal = Vector3();
	}
};

class Ray {
public:
	Vector3 Origin;
	Vector3 Direction;
	
	HOST_DEVICE FORCEINLINE Ray();
	HOST_DEVICE FORCEINLINE Ray(const Vector3& Origin, const Vector3& Direction) : Origin(Origin), Direction(Direction) { }
	HOST_DEVICE FORCEINLINE Vector3 GetOrigin() const { return Origin; }
	HOST_DEVICE FORCEINLINE Vector3 GetDirection() const { return Direction; }
	HOST_DEVICE FORCEINLINE Vector3 PointAt(float t) const { return Origin + (Direction * t); };
};

#include "Ray.inl"
