#pragma once

#include "Engine/CoreTypes.h"

namespace EmptySource {

	struct Vector3;
	struct Matrix4x4;
	struct Matrix3x3;

	enum AngleAxes {
		Pitch = 0, // up - down
		Yaw = 1,   // left - right
		Roll = 2   // fall over
	};

	struct Quaternion {
	public:
		union {
			struct { float w, x, y, z; };
		};

		HOST_DEVICE FORCEINLINE Quaternion();
		HOST_DEVICE FORCEINLINE Quaternion(Quaternion const& Other);
		HOST_DEVICE FORCEINLINE Quaternion(float const& Scalar, Vector3 const& Vector);
		HOST_DEVICE FORCEINLINE Quaternion(float const& w, float const& x, float const& y, float const& z);

		//* Create a quaternion from euler angles (pitch, yaw, roll).
		HOST_DEVICE static FORCEINLINE Quaternion EulerAngles(Vector3 const& EulerAngles);
		//* Create a quaternion from two normalized axis
		HOST_DEVICE static FORCEINLINE Quaternion FromToRotation(Vector3 const& From, Vector3 const& To);
		HOST_DEVICE static FORCEINLINE Quaternion AxisAngle(Vector3 const& Axis, float const& Degrees);
		//* Ctreate quaternion from two basis vectors
		HOST_DEVICE static FORCEINLINE Quaternion LookRotation(Vector3 const& Forward, Vector3 const& Up);
		HOST_DEVICE static FORCEINLINE Quaternion FromMatrix(Matrix3x3 const & Matrix);

		HOST_DEVICE inline float Magnitude() const;
		HOST_DEVICE inline float MagnitudeSquared() const;
		HOST_DEVICE inline void Normalize();
		HOST_DEVICE inline Quaternion Normalized() const;
		HOST_DEVICE inline Quaternion Conjugated() const;
		HOST_DEVICE inline Quaternion Inversed() const;

		HOST_DEVICE inline Matrix4x4 ToMatrix4x4() const;
		//* Deconstruct quaternion to euler angles pitch (degrees)
		HOST_DEVICE inline float GetPitch() const;
		//* Deconstruct quaternion to euler angles yaw (degrees)
		HOST_DEVICE inline float GetYaw() const;
		//* Deconstruct quaternion to euler angles roll (degrees)
		HOST_DEVICE inline float GetRoll() const;
		HOST_DEVICE inline float GetScalar() const;
		HOST_DEVICE inline Vector3 GetVector() const;
		//* Deconstruct quaternion to euler angles (degrees)
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
		HOST_DEVICE FORCEINLINE Vector3 operator*(const Vector3& Vector) const;

		HOST_DEVICE FORCEINLINE Quaternion& operator*=(const Quaternion& Other);
		HOST_DEVICE FORCEINLINE Quaternion& operator*=(const float& Value);
		HOST_DEVICE FORCEINLINE Quaternion& operator/=(const float& Value);
	};

}

#include "Math/Quaternion.inl"
