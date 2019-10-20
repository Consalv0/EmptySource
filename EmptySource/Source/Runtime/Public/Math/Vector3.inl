#pragma once

#include <math.h>
#include <stdexcept>
#include <string>

#include "Math/Vector2.h"
#include "Math/IntVector3.h"
#include "Math/Vector4.h"

namespace ESource {

	FORCEINLINE Vector3::Vector3()
		: x(0), y(0), z(0) {
	}

	FORCEINLINE Vector3::Vector3(const Vector2 & Vector)
		: x(Vector.x), y(Vector.y), z(0) {
	}

	FORCEINLINE Vector3::Vector3(const IntVector3 & Vector)
		: x(float(Vector.x)), y(float(Vector.y)), z(float(Vector.z)) {
	}

	FORCEINLINE Vector3::Vector3(const Vector3 & Vector)
		: x(Vector.x), y(Vector.y), z(Vector.z) {
	}

	FORCEINLINE Vector3::Vector3(const Vector4 & Vector)
		: x(Vector.x), y(Vector.y), z(Vector.z) {
	}

	FORCEINLINE Vector3::Vector3(const float& x, const float& y, const float& z)
		: x(x), y(y), z(z) {
	}

	FORCEINLINE Vector3::Vector3(const float& x, const float& y)
		: x(x), y(y), z(0) {
	}

	FORCEINLINE Vector3::Vector3(const float& Value)
		: x(Value), y(Value), z(Value) {
	}

	inline float Vector3::Magnitude() const {
		return sqrtf(x * x + y * y + z * z);
	}

	inline float Vector3::MagnitudeSquared() const {
		return x * x + y * y + z * z;
	}

	inline void Vector3::Normalize() {
		if (MagnitudeSquared() == 0) {
			x = 0; y = 0; z = 0;
		}
		else {
			*this /= Magnitude();
		}
	}

	inline Vector3 Vector3::Normalized() const {
		if (MagnitudeSquared() == 0) return Vector3();
		return *this / Magnitude();
	}

	FORCEINLINE Vector3 Vector3::Cross(const Vector3& Other) const {
		return Vector3(
			y * Other.z - z * Other.y,
			z * Other.x - x * Other.z,
			x * Other.y - y * Other.x
		);
	}

	inline Vector3 Vector3::Cross(const Vector3 & A, const Vector3 & B) {
		return Vector3(
			A.y * B.z - A.z * B.y,
			A.z * B.x - A.x * B.z,
			A.x * B.y - A.y * B.x
		);
	}

	FORCEINLINE float Vector3::Dot(const Vector3& Other) const {
		return (x * Other.x) + (y * Other.y) + (z * Other.z);
	}

	inline float Vector3::Dot(const Vector3 & A, const Vector3 & B) {
		return (A.x * B.x) + (A.y * B.y) + (A.z * B.z);
	}

	inline const float * Vector3::PointerToValue() const {
		return &x;
	}

	FORCEINLINE Vector3 Vector3::Lerp(const Vector3 & Start, const Vector3 & End, float t) {
		return Vector3((Start.x * (1.0F - t)) + (End.x * t), (Start.y * (1.0F - t)) + (End.y * t), (Start.z * (1.0F - t)) + (End.z * t));
	}

	inline HOST_DEVICE Vector3 Vector3::Reflect(const Vector3 & Incident, const Vector3 & Normal) {
		return Incident - (Normal * Normal.Dot(Incident)) * 2.F;
	}

	inline float & Vector3::operator[](unsigned char i) {
		ES_CORE_ASSERT(i <= 2, "Vector3 index out of bounds");
		return ((float*)this)[i];
	}

	inline float const & Vector3::operator[](unsigned char i) const {
		ES_CORE_ASSERT(i <= 2, "Vector3 index out of bounds");
		return ((float*)this)[i];
	}

	FORCEINLINE Vector3 Vector3::operator * (const Vector3& Other) const {
		return Vector3(x * Other.x, y * Other.y, z * Other.z);
	}

	FORCEINLINE Vector3 Vector3::operator/(const Vector3 & Other) const {
		return Vector3(x / Other.x, y / Other.y, z / Other.z);
	}

	FORCEINLINE bool Vector3::operator==(const Vector3& Other) const {
		return (x == Other.x && y == Other.y && z == Other.z);
	}

	FORCEINLINE bool Vector3::operator!=(const Vector3& Other) const {
		return (x != Other.x || y != Other.y || z != Other.z);
	}

	FORCEINLINE Vector3 Vector3::operator+(const Vector3& Other) const {
		return Vector3(x + Other.x, y + Other.y, z + Other.z);
	}

	FORCEINLINE Vector3 Vector3::operator-(const Vector3& Other) const {
		return Vector3(x - Other.x, y - Other.y, z - Other.z);
	}

	FORCEINLINE Vector3 Vector3::operator-(void) const {
		return Vector3(-x, -y, -z);
	}

	FORCEINLINE Vector3 Vector3::operator*(const float& Value) const {
		return Vector3(x * Value, y * Value, z * Value);
	}

	FORCEINLINE Vector3 Vector3::operator/(const float& Value) const {
		if (Value == 0) return Vector3();
		return Vector3(x / Value, y / Value, z / Value);
	}

	FORCEINLINE Vector3& Vector3::operator+=(const Vector3& Other) {
		x += Other.x; y += Other.y;	z += Other.z;
		return *this;
	}

	FORCEINLINE Vector3& Vector3::operator-=(const Vector3& Other) {
		x -= Other.x; y -= Other.y; z -= Other.z;
		return *this;
	}

	FORCEINLINE Vector3 & Vector3::operator*=(const Vector3 & Other) {
		x *= Other.x; y *= Other.y; z *= Other.z;
		return *this;
	}

	FORCEINLINE Vector3 & Vector3::operator/=(const Vector3 & Other) {
		x /= Other.x; y /= Other.y; z /= Other.z;
		return *this;
	}

	FORCEINLINE Vector3& Vector3::operator*=(const float& Value) {
		x *= Value; y *= Value; z *= Value;
		return *this;
	}

	FORCEINLINE Vector3& Vector3::operator/=(const float& Value) {
		if (Value == 0) x = y = z = 0;
		x /= Value; y /= Value; z /= Value;
		return *this;
	}

	inline Vector3 operator*(float Value, const Vector3 & Vector) {
		return Vector3(Value * Vector.x, Value * Vector.y, Value * Vector.z);
	}

	inline Vector3 operator/(float Value, const Vector3 & Vector) {
		return Vector3(Value / Vector.x, Value / Vector.y, Value / Vector.z);
	}

	inline Vector3 Math::NormalizeAngleComponents(Vector3 EulerAngle) {
		EulerAngle.x = NormalizeAngle(EulerAngle.x);
		EulerAngle.y = NormalizeAngle(EulerAngle.y);
		EulerAngle.z = NormalizeAngle(EulerAngle.z);

		return EulerAngle;
	}

	inline Vector3 Math::ClampAngleComponents(Vector3 EulerAngle) {
		EulerAngle.x = ClampAngle(EulerAngle.x);
		EulerAngle.y = ClampAngle(EulerAngle.y);
		EulerAngle.z = ClampAngle(EulerAngle.z);

		return EulerAngle;
	}

}