#include "MathUtility.h"
#pragma once

namespace Math {
	template <class T>
	inline T Max(const T& A, const T& B) { return (((A) > (B)) ? (A) : (B)); }

	template <class T>
	inline T Min(const T& A, const T& B) { return (((A) < (B)) ? (A) : (B)); }
	
	//* The number is power of 2
	inline int IsPow2(const int& x) {
		return ((x&(x - 1)) == 0);
	}

	//* Get the next power2 of the value
	inline unsigned int NextPow2(unsigned int x) {
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}
}

float Math::ClampAngle(float Angle) {
	Angle = std::fmod(Angle, 360.F);

	if (Angle < 0.F) {
		// --- Shift to [0,360)
		Angle += 360.F;
	}

	return Angle;
}

float Math::NormalizeAngle(float Angle) {
	Angle = ClampAngle(Angle);

	if (Angle > 180.f) {
		// --- Shift to (-180,180]
		Angle -= 360.f;
	}

	return Angle;
}

Vector3 Math::NormalizeAngleComponents(Vector3 EulerAngle) {
	EulerAngle.x = NormalizeAngle(EulerAngle.x);
	EulerAngle.y = NormalizeAngle(EulerAngle.y);
	EulerAngle.z = NormalizeAngle(EulerAngle.z);

	return EulerAngle;
}

Vector3 Math::ClampAngleComponents(Vector3 EulerAngle) {
	EulerAngle.x = ClampAngle(EulerAngle.x);
	EulerAngle.y = ClampAngle(EulerAngle.y);
	EulerAngle.z = ClampAngle(EulerAngle.z);

	return EulerAngle;
}

float Math::Pow10(int Number) {
	float Ret = 1.0F;
	float R = 10.0F;
	if (Number < 0) {
		Number = -Number;
		R = 0.1F;
	}

	while (Number) {
		if (Number & 1) {
			Ret *= R;
		}
		R *= R;
		Number >>= 1;
	}
	return Ret;
}
