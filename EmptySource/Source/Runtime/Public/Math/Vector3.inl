#pragma once

#include <math.h>
#include <stdexcept>
#include <string>

#include "Math/Vector2.h"
#include "Math/IntVector3.h"
#include "Math/Vector4.h"

namespace ESource {

	FORCEINLINE Vector3::Vector3()
		: X(0), Y(0), Z(0) {
	}

	FORCEINLINE Vector3::Vector3(const Vector2 & Vector)
		: X(Vector.X), Y(Vector.Y), Z(0) {
	}

	FORCEINLINE Vector3::Vector3(const IntVector3 & Vector)
		: X(float(Vector.X)), Y(float(Vector.Y)), Z(float(Vector.Z)) {
	}

	FORCEINLINE Vector3::Vector3(const Vector3 & Vector)
		: X(Vector.X), Y(Vector.Y), Z(Vector.Z) {
	}

	FORCEINLINE Vector3::Vector3(const Vector4 & Vector)
		: X(Vector.X), Y(Vector.Y), Z(Vector.Z) {
	}

	FORCEINLINE Vector3::Vector3(const float& X, const float& Y, const float& Z)
		: X(X), Y(Y), Z(Z) {
	}

	FORCEINLINE Vector3::Vector3(const float& X, const float& Y)
		: X(X), Y(Y), Z(0) {
	}

	FORCEINLINE Vector3::Vector3(const float& Value)
		: X(Value), Y(Value), Z(Value) {
	}

	inline float Vector3::Magnitude() const {
		return sqrtf(X * X + Y * Y + Z * Z);
	}

	inline float Vector3::MagnitudeSquared() const {
		return X * X + Y * Y + Z * Z;
	}

	inline void Vector3::Normalize() {
		if (MagnitudeSquared() == 0) {
			X = 0; Y = 0; Z = 0;
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
			Y * Other.Z - Z * Other.Y,
			Z * Other.X - X * Other.Z,
			X * Other.Y - Y * Other.X
		);
	}

	inline Vector3 Vector3::Cross(const Vector3 & A, const Vector3 & B) {
		return Vector3(
			A.Y * B.Z - A.Z * B.Y,
			A.Z * B.X - A.X * B.Z,
			A.X * B.Y - A.Y * B.X
		);
	}

	FORCEINLINE float Vector3::Dot(const Vector3& Other) const {
		return (X * Other.X) + (Y * Other.Y) + (Z * Other.Z);
	}

	inline float Vector3::Dot(const Vector3 & A, const Vector3 & B) {
		return (A.X * B.X) + (A.Y * B.Y) + (A.Z * B.Z);
	}

	inline const float * Vector3::PointerToValue() const {
		return &X;
	}

	FORCEINLINE Vector3 Vector3::Lerp(const Vector3 & Start, const Vector3 & End, float t) {
		return Vector3((Start.X * (1.0F - t)) + (End.X * t), (Start.Y * (1.0F - t)) + (End.Y * t), (Start.Z * (1.0F - t)) + (End.Z * t));
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
		return Vector3(X * Other.X, Y * Other.Y, Z * Other.Z);
	}

	FORCEINLINE Vector3 Vector3::operator/(const Vector3 & Other) const {
		return Vector3(X / Other.X, Y / Other.Y, Z / Other.Z);
	}

	FORCEINLINE bool Vector3::operator==(const Vector3& Other) const {
		return (X == Other.X && Y == Other.Y && Z == Other.Z);
	}

	FORCEINLINE bool Vector3::operator!=(const Vector3& Other) const {
		return (X != Other.X || Y != Other.Y || Z != Other.Z);
	}

	FORCEINLINE Vector3 Vector3::operator+(const Vector3& Other) const {
		return Vector3(X + Other.X, Y + Other.Y, Z + Other.Z);
	}

	FORCEINLINE Vector3 Vector3::operator-(const Vector3& Other) const {
		return Vector3(X - Other.X, Y - Other.Y, Z - Other.Z);
	}

	FORCEINLINE Vector3 Vector3::operator-(void) const {
		return Vector3(-X, -Y, -Z);
	}

	FORCEINLINE Vector3 Vector3::operator*(const float& Value) const {
		return Vector3(X * Value, Y * Value, Z * Value);
	}

	FORCEINLINE Vector3 Vector3::operator/(const float& Value) const {
		if (Value == 0) return Vector3();
		return Vector3(X / Value, Y / Value, Z / Value);
	}

	FORCEINLINE Vector3& Vector3::operator+=(const Vector3& Other) {
		X += Other.X; Y += Other.Y;	Z += Other.Z;
		return *this;
	}

	FORCEINLINE Vector3& Vector3::operator-=(const Vector3& Other) {
		X -= Other.X; Y -= Other.Y; Z -= Other.Z;
		return *this;
	}

	FORCEINLINE Vector3 & Vector3::operator*=(const Vector3 & Other) {
		X *= Other.X; Y *= Other.Y; Z *= Other.Z;
		return *this;
	}

	FORCEINLINE Vector3 & Vector3::operator/=(const Vector3 & Other) {
		X /= Other.X; Y /= Other.Y; Z /= Other.Z;
		return *this;
	}

	FORCEINLINE Vector3& Vector3::operator*=(const float& Value) {
		X *= Value; Y *= Value; Z *= Value;
		return *this;
	}

	FORCEINLINE Vector3& Vector3::operator/=(const float& Value) {
		if (Value == 0) X = Y = Z = 0;
		X /= Value; Y /= Value; Z /= Value;
		return *this;
	}

	inline Vector3 operator*(float Value, const Vector3 & Vector) {
		return Vector3(Value * Vector.X, Value * Vector.Y, Value * Vector.Z);
	}

	inline Vector3 operator/(float Value, const Vector3 & Vector) {
		return Vector3(Value / Vector.X, Value / Vector.Y, Value / Vector.Z);
	}

}

inline ESource::Vector3 Math::NormalizeAngleComponents(ESource::Vector3 EulerAngle) {
	EulerAngle.X = NormalizeAngle(EulerAngle.X);
	EulerAngle.Y = NormalizeAngle(EulerAngle.Y);
	EulerAngle.Z = NormalizeAngle(EulerAngle.Z);

	return EulerAngle;
}

inline ESource::Vector3 Math::ClampAngleComponents(ESource::Vector3 EulerAngle) {
	EulerAngle.X = ClampAngle(EulerAngle.X);
	EulerAngle.Y = ClampAngle(EulerAngle.Y);
	EulerAngle.Z = ClampAngle(EulerAngle.Z);

	return EulerAngle;
}