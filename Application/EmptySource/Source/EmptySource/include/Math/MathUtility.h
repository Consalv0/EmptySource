#pragma once

#include <math.h>
#include <algorithm>

struct Vector3;

namespace MathConstants {
	static const float Pi = 3.1415926535897932F;
	static const float SmallNumber = 1.e-8F;
	static const float Big_Number = 3.4e+38f;
	static const float Euler = 2.71828182845904523536F;
	static const float InversePi = 0.31830988618F;
	static const float HalfPi = 1.57079632679F;
	static const float DeltaPresicion = 0.00001F;
	static const float RadToDegree = 57.2957795130823208F;
	static const float DegreeToRad = 0.0174532925199432F;
}

namespace Math {
	//* Get the angle in degrees in range of [0, 360)
	inline float ClampAngle(float Angle);

	//* Get the angle in degrees in the range of (-180, 180]
	inline float NormalizeAngle(float Angle);

	//* Get the angles in degrees in the range of (-180, 180) 
	inline Vector3 NormalizeAngleComponents(Vector3 EulerAngle);

	//* Get the angles in degrees in the range of [0, 360)
	inline Vector3 ClampAngleComponents(Vector3 EulerAngle);

	//* Fast pow to ten
	inline float Pow10(int Number);
}
