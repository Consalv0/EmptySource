#pragma once

#include "CoreTypes.h"

namespace ESource {

	struct IntVector3;
	struct Vector2;
	struct Vector4;

	struct Vector3 {
	public:
		union {
			struct { float X, Y, Z; };
			struct { float R, G, B; };
		};

		HOST_DEVICE FORCEINLINE Vector3();
		HOST_DEVICE FORCEINLINE Vector3(const Vector2& Vector);
		HOST_DEVICE FORCEINLINE Vector3(const IntVector3& Vector);
		HOST_DEVICE FORCEINLINE Vector3(const Vector3& Vector);
		HOST_DEVICE FORCEINLINE Vector3(const Vector4& Vector);
		HOST_DEVICE FORCEINLINE Vector3(const float& Value);
		HOST_DEVICE FORCEINLINE Vector3(const float& X, const float& Y, const float& Z);
		HOST_DEVICE FORCEINLINE Vector3(const float& X, const float& Y);

		HOST_DEVICE inline float Magnitude() const;
		HOST_DEVICE inline float MagnitudeSquared() const;
		HOST_DEVICE inline void Normalize();
		HOST_DEVICE inline Vector3 Normalized() const;

		HOST_DEVICE FORCEINLINE Vector3 Cross(const Vector3& Other) const;
		HOST_DEVICE FORCEINLINE static Vector3 Cross(const Vector3 &A, const Vector3 &B);
		HOST_DEVICE FORCEINLINE float Dot(const Vector3& Other) const;
		HOST_DEVICE FORCEINLINE static float Dot(const Vector3 &A, const Vector3 &B);
		HOST_DEVICE FORCEINLINE static Vector3 Lerp(const Vector3& start, const Vector3& end, float t);
		HOST_DEVICE FORCEINLINE static Vector3 Reflect(const Vector3& Incident, const Vector3& Normal);

		HOST_DEVICE inline float & operator[](unsigned char i);
		HOST_DEVICE inline float const& operator[](unsigned char i) const;
		HOST_DEVICE inline const float* PointerToValue() const;

		HOST_DEVICE FORCEINLINE bool operator==(const Vector3& Other) const;
		HOST_DEVICE FORCEINLINE bool operator!=(const Vector3& Other) const;

		HOST_DEVICE FORCEINLINE Vector3 operator+(const Vector3& Other) const;
		HOST_DEVICE FORCEINLINE Vector3 operator-(const Vector3& Other) const;
		HOST_DEVICE FORCEINLINE Vector3 operator-(void) const;
		HOST_DEVICE FORCEINLINE Vector3 operator*(const Vector3& Other) const;
		HOST_DEVICE FORCEINLINE Vector3 operator/(const Vector3& Other) const;
		HOST_DEVICE FORCEINLINE Vector3 operator*(const float& Value) const;
		HOST_DEVICE FORCEINLINE Vector3 operator/(const float& Value) const;

		HOST_DEVICE FORCEINLINE Vector3& operator+=(const Vector3& Other);
		HOST_DEVICE FORCEINLINE Vector3& operator-=(const Vector3& Other);
		HOST_DEVICE FORCEINLINE Vector3& operator*=(const Vector3& Other);
		HOST_DEVICE FORCEINLINE Vector3& operator/=(const Vector3& Other);
		HOST_DEVICE FORCEINLINE Vector3& operator*=(const float& Value);
		HOST_DEVICE FORCEINLINE Vector3& operator/=(const float& Value);

		HOST_DEVICE inline friend Vector3 operator*(float Value, const Vector3 &Vector);
		HOST_DEVICE inline friend Vector3 operator/(float Value, const Vector3 &Vector);
	};

	typedef Vector3 Point3;

}

namespace Math {
	//* Get the angles in degrees in the range of (-180, 180) 
	inline ESource::Vector3 NormalizeAngleComponents(ESource::Vector3 EulerAngle);

	//* Get the angles in degrees in the range of [0, 360)
	inline ESource::Vector3 ClampAngleComponents(ESource::Vector3 EulerAngle);
}

#include "Math/Vector3.inl"
