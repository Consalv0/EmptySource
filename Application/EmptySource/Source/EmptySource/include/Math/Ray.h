#pragma once

#include "Vector3.h"

class Ray {
public:
	Vector3 A;
	Vector3 B;
	
	Ray() {}
	Ray(const Vector3& a, const Vector3& b) { A = a; B = b; }
	Vector3 Origin() const { return A; }
	Vector3 Direction() const { return B; }
	Vector3 PointAt(float t) const { return A + B * t; };
	
};

#include "Ray.inl"