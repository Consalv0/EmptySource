#pragma once

#include "Vector3.h"

class Ray {
public:
	Vector3 A;
	Vector3 B;
	
	HOST_DEVICE FORCEINLINE Ray();
	HOST_DEVICE FORCEINLINE Ray(const Vector3& a, const Vector3& b) { A = a; B = b; }
	HOST_DEVICE FORCEINLINE Vector3 Origin() const { return A; }
	HOST_DEVICE FORCEINLINE Vector3 Direction() const { return B; }
	HOST_DEVICE FORCEINLINE Vector3 PointAt(float t) const { return A + B * t; };
};

#include "Ray.inl"