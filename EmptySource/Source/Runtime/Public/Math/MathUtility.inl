
#include "Math/MathUtility.h"

namespace ESource {

	namespace MathEquations {

		int SolveQuadratic(float X[2], float A, float B, float C) {
			if (fabs(A) < 1e-14F) {
				if (fabs(B) < 1e-14F) {
					if (C == 0)
						return -1;
					return 0;
				}
				X[0] = -C / B;
				return 1;
			}
			float dscr = B * B - 4 * A*C;
			if (dscr > 0) {
				dscr = sqrtf(dscr);
				X[0] = (-B + dscr) / (2 * A);
				X[1] = (-B - dscr) / (2 * A);
				return 2;
			}
			else if (dscr == 0) {
				X[0] = -B / (2 * A);
				return 1;
			}
			else
				return 0;
		}

		int SolveCubicNormed(float * X, float A, float B, float C) {
			float A2 = A * A;
			float Q = (A2 - 3.F * B) / 9.F;
			float R = (A*(2.F * A2 - 9.F * B) + 27.F * C) / 54.F;
			float R2 = R * R;
			float Q3 = Q*Q*Q;
			if (R2 < Q3) {
				float t = R / sqrtf(Q3);
				if (t < -1.F) t = -1.F;
				if (t > 1.F) t = 1.F;
				t = acosf(t);
				A /= 3.F; Q = -2.F * sqrtf(Q);
				X[0] = Q * cosf(t / 3.F) - A;
				X[1] = Q * cosf((t + 2.F * MathConstants::Pi) / 3.F) - A;
				X[2] = Q * cosf((t - 2.F * MathConstants::Pi) / 3.F) - A;
				return 3;
			}
			else {
				float A1, B1;
				A1 = -powf(fabs(R) + sqrtf(R2 - Q3), 1 / 3.F);
				if (R < 0) A1 = -A1;
				B1 = A1 == 0 ? 0 : Q / A1;
				A /= 3;
				X[0] = (A1 + B1) - A;
				X[1] = -0.5F*(A1 + B1) - A;
				X[2] = 0.5F*sqrtf(3.F)*(A1 - B1);
				if (fabs(X[2]) < 1e-14F)
					return 2;
				return 1;
			}
		}

		int SolveCubic(float X[3], float A, float B, float C, float D) {
			if (fabs(A) < 1e-14F)
				return SolveQuadratic(X, B, C, D);
			return SolveCubicNormed(X, B / A, C / A, D / A);
		}

		template<typename T>
		float Shoelace2(const T & A, const T & B) {
			return (B[0] - A[0]) * (A[1] + B[1]);
		}
	}

	namespace Math {
		//* The number is power of 2
		inline int IsPow2(const int A) {
			return ((A & (A - 1)) == 0);
		}

		//* Get the next power2 of the value
		inline int NextPow2(int X) {
			--X;
			X |= X >> 1;
			X |= X >> 2;
			X |= X >> 4;
			X |= X >> 8;
			X |= X >> 16;
			return ++X;
		}
	}

	template <typename T>
	T Math::Min(const T A, const T B) {
		return B < A ? B : A;
	}

	template <typename T>
	T Math::Max(const T A, const T B) {
		return A < B ? B : A;
	}

	template <typename T, typename S>
	T Math::Median(const T A, const T B, const S Alpha) {
		return Max(Min(A, B), Min(Max(A, B), Alpha));
	}

	template <typename T, typename S>
	T Math::Mix(const T A, const T B, const S Weight) {
		return T((S(1) - Weight) * A + Weight * B);
	}

	template<typename T>
	T Math::Abs(const T Value) {
		return fabs(Value);
	}

	template<typename T>
	T Math::Sign(const T Value) {
		return (T(0) < Value) - (Value < T(0));
	}

	template<typename T>
	T Math::NonZeroSign(const T Value) {
		return T(2) * (Value > T(0)) - T(1);
	}

	template<typename T>
	T Math::Square(const T Value) {
		return Value * Value;
	}

	template <typename T>
	T Math::Clamp(const T Value, const T A) {
		return Value >= T(0) && Value <= A ? Value : T(Value > T(0)) * A;
	}

	template <typename T>
	T Math::Clamp(const T Value, const T A, const T B) {
		return Value >= A && Value <= B ? Value : Value < A ? A : B;
	}

	template <typename T>
	T Math::Clamp01(const T Value) {
		return Value >= T(0) && Value <= T(1) ? Value : Value < T(0) ? T(0) : T(1);
	}

	float Math::ClampAngle(float Degrees) {
		Degrees = fmod(Degrees, 360.F);

		if (Degrees < 0.F) {
			// --- Shift to [0,360)
			Degrees += 360.F;
		}

		return Degrees;
	}

	float Math::NormalizeAngle(float Degrees) {
		Degrees = ClampAngle(Degrees);

		if (Degrees > 180.f) {
			// --- Shift to (-180,180]
			Degrees -= 360.f;
		}

		return Degrees;
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

	// Referenced from UE4 implementation
	float Math::Atan2(float Y, float X) {
		const float absX = Math::Abs(X);
		const float absY = Math::Abs(Y);
		const bool yAbsBigger = (absY > absX);
		float t0 = yAbsBigger ? absY : absX; // Max(absY, absX)
		float t1 = yAbsBigger ? absX : absY; // Min(absX, absY)

		if (t0 == 0.F)
			return 0.F;

		float t3 = t1 / t0;
		float t4 = t3 * t3;

		static const float c[7] = {
			+7.2128853633444123e-03F,
			-3.5059680836411644e-02F,
			+8.1675882859940430e-02F,
			-1.3374657325451267e-01F,
			+1.9856563505717162e-01F,
			-3.3324998579202170e-01F,
			+1.0F
		};

		t0 = c[0];
		t0 = t0 * t4 + c[1];
		t0 = t0 * t4 + c[2];
		t0 = t0 * t4 + c[3];
		t0 = t0 * t4 + c[4];
		t0 = t0 * t4 + c[5];
		t0 = t0 * t4 + c[6];
		t3 = t0 * t3;

		t3 = yAbsBigger ? (0.5F * MathConstants::Pi) - t3 : t3;
		t3 = (X < 0.0F) ? MathConstants::Pi - t3 : t3;
		t3 = (Y < 0.0F) ? -t3 : t3;

		return t3;
	}

}