#pragma once

#include "../CoreTypes.h"

struct IntVector3;
struct Vector2;
struct Vector4;

struct Vector3 {
public:
	union {
		struct { float x, y, z; };
		struct { float r, g, b; };
	};

	HOST_DEVICE FORCEINLINE Vector3();
	HOST_DEVICE FORCEINLINE Vector3(const Vector2& Vector);
	HOST_DEVICE FORCEINLINE Vector3(const IntVector3& Vector);
	HOST_DEVICE FORCEINLINE Vector3(const Vector3& Vector);
	HOST_DEVICE FORCEINLINE Vector3(const Vector4& Vector);
	HOST_DEVICE FORCEINLINE Vector3(const float& Value);
	HOST_DEVICE FORCEINLINE Vector3(const float& x, const float& y, const float& z);
	HOST_DEVICE FORCEINLINE Vector3(const float& x, const float& y);

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

	HOST_DEVICE inline float & operator[](unsigned int i);
	HOST_DEVICE inline float const& operator[](unsigned int i) const;
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

#include "../Math/Vector3.inl"
