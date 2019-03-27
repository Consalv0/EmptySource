#pragma once

#include "../CoreTypes.h"

struct Vector3;
struct Matrix4x4;

struct Quaternion {
public:
	union {
		struct { float x, y, z, w; };
	};

	HOST_DEVICE FORCEINLINE Quaternion();
	HOST_DEVICE FORCEINLINE Quaternion(Quaternion const& Other);
	HOST_DEVICE FORCEINLINE Quaternion(Vector3 const& Axis, float const& Angle);
	HOST_DEVICE FORCEINLINE Quaternion(float const& Scalar, Vector3 const& Vector);
	HOST_DEVICE FORCEINLINE Quaternion(float const& w, float const& x, float const& y, float const& z);
	// Create a quaternion from two normalized axis
	HOST_DEVICE FORCEINLINE Quaternion(Vector3 const& u, Vector3 const& v);
	// Create a quaternion from euler angles (pitch, yaw, roll), in radians.
	HOST_DEVICE FORCEINLINE Quaternion(Vector3 const& EulerAngles);

	HOST_DEVICE inline float Magnitude() const;
	HOST_DEVICE inline float MagnitudeSquared() const;
	HOST_DEVICE inline void Normalize();
	HOST_DEVICE inline Quaternion Normalized() const;
	HOST_DEVICE inline Quaternion Conjugated() const;
	HOST_DEVICE inline Quaternion Inversed() const;

	HOST_DEVICE inline Matrix4x4 ToMatrix4x4() const;
	HOST_DEVICE inline float GetPitch() const;
	HOST_DEVICE inline float GetYaw() const;
	HOST_DEVICE inline float GetRoll() const;
	HOST_DEVICE inline float GetScalar() const;
	HOST_DEVICE inline Vector3 GetVector() const;
	HOST_DEVICE inline Vector3 ToEulerAngles() const;

	HOST_DEVICE FORCEINLINE float Dot(const Quaternion& Other) const;
	HOST_DEVICE FORCEINLINE Quaternion Cross(const Quaternion& Other) const;

	HOST_DEVICE inline float & operator[](unsigned int i);
	HOST_DEVICE inline float const& operator[](unsigned int i) const;
	HOST_DEVICE inline const float* PointerToValue() const;

	HOST_DEVICE FORCEINLINE bool operator==(const Quaternion& Other) const;
	HOST_DEVICE FORCEINLINE bool operator!=(const Quaternion& Other) const;
	
	HOST_DEVICE FORCEINLINE Quaternion operator-(void) const;
	HOST_DEVICE FORCEINLINE Quaternion operator*(const float& Value) const;
	HOST_DEVICE FORCEINLINE Quaternion operator/(const float& Value) const;
	HOST_DEVICE FORCEINLINE Quaternion operator*(const Quaternion& Other) const;
	
	HOST_DEVICE FORCEINLINE Quaternion& operator*=(const Quaternion& Other);
	HOST_DEVICE FORCEINLINE Quaternion& operator*=(const float& Value);
	HOST_DEVICE FORCEINLINE Quaternion& operator/=(const float& Value);
};

#include "../Math/Quaternion.inl"
