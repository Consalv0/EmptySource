#pragma once

#include <math.h>
#include <algorithm>

namespace MathConstants {
	static const float Pi = 3.1415926535897932F;
	static const float SmallNumber = 1.e-8F;
	static const float BigNumber = 3.4e+38f;
	static const float Euler = 2.71828182845904523536F;
	static const float InversePi = 0.31830988618F;
	static const float HalfPi = 1.57079632679F;
	static const float DeltaPresicion = 0.00001F;
	static const float RadToDegree = 57.2957795130823208F;
	static const float DegreeToRad = 0.0174532925199432F;
	static const float SquareRoot2 = 1.41421356237F;
}

namespace MathEquations {
	inline int SolveQuadratic(float x[2], float a, float b, float c);

	inline int SolveCubicNormed(float *x, float a, float b, float c);

	inline int SolveCubic(float x[3], float a, float b, float c, float d);

	template <typename T>
	inline float Shoelace2(const T & A, const T & B);
}

namespace Math {
	//* Returns the smaller of the arguments.
	template <typename T>
	inline T Min(const T & A, const T & B);

	//* Returns the larger of the arguments.
	template <typename T>
	inline T Max(const T & A, const T & B);

	//* Returns the middle out of three values
	template <typename T, typename S>
	inline T Median(T A, T B, S Alpha);

	//* Returns the weighted average of a and b.
	template <typename T, typename S>
	inline T Mix(T A, T B, S Weight);

	//* Returns 1 for positive values, -1 for negative values, and 0 for zero.
	template <typename T>
	inline int Sign(T Value);

	/// Returns 1 for non-negative values and -1 for negative values.
	template <typename T>
	inline int NonZeroSign(T Value);

	//* Clamps the number to the interval from 0 to b.
	template <typename T>
	inline T Clamp(T Value, const T & A);

	//* Clamp the value in the defined range 
	template <typename T>
	inline T Clamp(T Value, const T& A, const T& B);

	//* Clamp the value in range of [0, 1] 
	template <typename T>
	inline T Clamp01(T Value);

	//* Get the angle in degrees in range of [0, 360)
	inline float ClampAngle(float Degrees);

	//* Get the angle in degrees in the range of (-180, 180]
	inline float NormalizeAngle(float Degrees);

	//* Fast pow to ten
	inline float Pow10(int Number);
}

#include "MathUtility.inl"