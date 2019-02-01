#include "MathUtility.h"
#pragma once

float Math::ClampAngle(float Angle) {
	Angle = std::fmod(Angle, 360.F);

	if (Angle < 0.F) {
		// Shift to [0,360)
		Angle += 360.F;
	}

	return Angle;
}

float Math::NormalizeAngle(float Angle) {
	Angle = ClampAngle(Angle);

	if (Angle > 180.f) {
		// Shift to (-180,180]
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
