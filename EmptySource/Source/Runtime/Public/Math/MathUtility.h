#pragma once

#include <math.h>
#include <algorithm>

namespace MathConstants {
	static constexpr float Pi = 3.1415926535897932F;
	static constexpr float TendencyZero = 1e-6F;
	static constexpr float SmallNumber = 1.e-8F;
	static constexpr float BigNumber = 3.4e+38f;
	static constexpr float Euler = 2.71828182845904523536F;
	static constexpr float InversePi = 0.31830988618F;
	static constexpr float HalfPi = 1.57079632679F;
	static constexpr float DeltaPresicion = 0.00001F;
	static constexpr float RadToDegree = 57.2957795130823208F;
	static constexpr float DegreeToRad = 0.0174532925199432F;
	static constexpr float SquareRoot2 = 1.41421356237F;
}

namespace MathEquations {
	inline int SolveQuadratic(float X[2], float A, float B, float C);

	inline int SolveCubicNormed(float *X, float A, float B, float C);

	inline int SolveCubic(float X[3], float A, float B, float C, float D);

	template <typename T>
	inline float Shoelace2(const T & A, const T & B);
}

namespace Math {
	//* Returns the smaller of the arguments.
	template <typename T>
	inline T Min(const T A, const T B);

	//* Returns the larger of the arguments.
	template <typename T>
	inline T Max(const T A, const T B);

	//* Returns the middle out of three values
	template <typename T, typename S>
	inline T Median(const T A, const T B, const S Alpha);

	//* Returns the weighted average of a and b.
	template <typename T, typename S>
	inline T Mix(const T A, const T B, const S Weight);

	//* Get the absolute value
	template <typename T>
	inline T Abs(const T Value);

	//* Returns 1 for positive values, -1 for negative values, and 0 for zero.
	template <typename T>
	inline T Sign(const T Value);

	/// Returns 1 for non-negative values and -1 for negative values.
	template <typename T>
	inline T NonZeroSign(const T Value);

	template <typename T>
	inline T Square(const T Value);

	//* Remap the value to another range of values
	template <typename T>
	inline T Map(const T Value, const T MinA, const T MaxA, const T MinB, const T MaxB);

	//* Clamps the number to the interval from 0 to b.
	template <typename T>
	inline T Clamp(const T Value, const T A);

	//* Clamp the value in the defined range 
	template <typename T>
	inline T Clamp(const T Value, const T A, const T B);

	//* Clamp the value in range of [0, 1] 
	template <typename T>
	inline T Clamp01(const T Value);

	//* Get the angle in degrees in range of [0, 360)
	inline float ClampAngle(float Degrees);

	//* Get the angle in degrees in the range of (-180, 180]
	inline float NormalizeAngle(float Degrees);

	//* Fast pow to ten
	inline float Pow10(int Number);

	//* Error consideration
	inline float Atan2(float Y, float X);
}

#include "Math/MathUtility.inl"
