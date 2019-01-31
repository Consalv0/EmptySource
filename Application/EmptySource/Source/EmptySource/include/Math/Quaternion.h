#pragma once

#include "..\CoreTypes.h"

struct Vector3;
struct Matrix4x4;

struct Quaternion {
public:
	union {
		struct { float x, y, z, w; };
	};

	FORCEINLINE Quaternion();
	FORCEINLINE Quaternion(Quaternion const& Other);
	FORCEINLINE Quaternion(float const& Angle, Vector3 const& Axis);
	FORCEINLINE Quaternion(float const& w, float const& x, float const& y, float const& z);
	// Create a quaternion from two normalized axis
	FORCEINLINE Quaternion(Vector3 const& u, Vector3 const& v);
	// Create a quaternion from euler angles (pitch, yaw, roll), in radians.
	FORCEINLINE Quaternion(Vector3 const& Angles);

	inline float Magnitude() const;
	inline float MagnitudeSquared() const;
	inline void Normalize();
	inline Quaternion Normalized() const;

	inline Matrix4x4 ToMatrix4x4() const;

	FORCEINLINE float Dot(const Quaternion& Other) const;
	FORCEINLINE Quaternion Cross(const Quaternion& Other) const;

	inline float & operator[](unsigned int i);
	inline float const& operator[](unsigned int i) const;
	inline const float* PointerToValue() const;

	FORCEINLINE bool operator==(const Quaternion& Other) const;
	FORCEINLINE bool operator!=(const Quaternion& Other) const;

	FORCEINLINE Quaternion operator-(void) const;
	FORCEINLINE Quaternion operator*(const float& Value) const;
	FORCEINLINE Quaternion operator/(const float& Value) const;
	FORCEINLINE Quaternion operator*(const Quaternion& Other) const;

	FORCEINLINE Quaternion& operator*=(const Quaternion& Other);
	FORCEINLINE Quaternion& operator*=(const float& Value);
	FORCEINLINE Quaternion& operator/=(const float& Value);

	inline WString ToString();
};

#include "..\Math\Quaternion.inl"