#include "MathUtility.h"

namespace MathEquations {
	int SolveQuadratic(float x[2], float a, float b, float c) {
		if (fabs(a) < 1e-14F) {
			if (fabs(b) < 1e-14F) {
				if (c == 0)
					return -1;
				return 0;
			}
			x[0] = -c / b;
			return 1;
		}
		float dscr = b * b - 4 * a*c;
		if (dscr > 0) {
			dscr = sqrtf(dscr);
			x[0] = (-b + dscr) / (2 * a);
			x[1] = (-b - dscr) / (2 * a);
			return 2;
		}
		else if (dscr == 0) {
			x[0] = -b / (2 * a);
			return 1;
		}
		else
			return 0;
	}

	int SolveCubicNormed(float * x, float a, float b, float c) {
		float a2 = a * a;
		float q = (a2 - 3 * b) / 9;
		float r = (a*(2 * a2 - 9 * b) + 27 * c) / 54;
		float r2 = r * r;
		float q3 = q * q*q;
		float A, B;
		if (r2 < q3) {
			float t = r / sqrtf(q3);
			if (t < -1) t = -1;
			if (t > 1) t = 1;
			t = acosf(t);
			a /= 3; q = -2 * sqrtf(q);
			x[0] = q * cosf(t / 3) - a;
			x[1] = q * cosf((t + 2 * MathConstants::Pi) / 3) - a;
			x[2] = q * cosf((t - 2 * MathConstants::Pi) / 3) - a;
			return 3;
		}
		else {
			A = -powf(fabs(r) + sqrtf(r2 - q3), 1 / 3.F);
			if (r < 0) A = -A;
			B = A == 0 ? 0 : q / A;
			a /= 3;
			x[0] = (A + B) - a;
			x[1] = -0.5F*(A + B) - a;
			x[2] = 0.5F*sqrtf(3.F)*(A - B);
			if (fabs(x[2]) < 1e-14F)
				return 2;
			return 1;
		}
	}

	int SolveCubic(float x[3], float a, float b, float c, float d) {
		if (fabs(a) < 1e-14F)
			return SolveQuadratic(x, b, c, d);
		return SolveCubicNormed(x, b / a, c / a, d / a);
	}
}

namespace Math {
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

template <typename T>
T Math::Min(const T & A, const T & B) {
	return B < A ? B : A;
}

template <typename T>
T Math::Max(const T & A, const T & B) {
	return A < B ? B : A;
}

template <typename T, typename S>
T Math::Median(T A, T B, S Alpha) {
	return Max(Min(A, B), Min(Max(A, B), Alpha));
}

template <typename T, typename S>
T Math::Mix(T A, T B, S Weight) {
	return T((S(1) - Weight) * A + Weight * B);
}

template<typename T>
int Math::Sign(T Value) {
	return (0.F < Value) - (Value < 0.F);
}

template<typename T>
int Math::NonZeroSign(T Value) {
	return 2 * (Value > T(0)) - 1;
}

template <typename T>
T Math::Clamp(T Value, const T & A) {
	return Value >= T(0) && Value <= A ? Value : T(Value > T(0)) * A;
}

template <typename T>
T Math::Clamp(T Value, const T & A, const T & B) {
	return Value >= A && Value <= B ? Value : Value < A ? A : B;
}

template <typename T>
T Math::Clamp01(T Value) {
	if (Value < T(0)) Value = T(0);
	else if (Value > T(1)) Value = T(1);
	return Value;
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
