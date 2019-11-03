#pragma once

#include <math.h>
#include <stdexcept>

#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/Vector2.h"

namespace ESource {

	FORCEINLINE Vector2::Vector2()
		: X(0), Y(0) {
	}

	FORCEINLINE Vector2::Vector2(const Vector2 & Vector)
		: X(Vector.X), Y(Vector.Y) {
	}

	FORCEINLINE Vector2::Vector2(const Vector3 & Vector)
		: X(Vector.X), Y(Vector.Y) {
	}

	FORCEINLINE Vector2::Vector2(const Vector4 & Vector)
		: X(Vector.X), Y(Vector.Y) {
	}

	FORCEINLINE Vector2::Vector2(const float& X, const float& Y)
		: X(X), Y(Y) {
	}

	FORCEINLINE Vector2::Vector2(const float& Value)
		: X(Value), Y(Value) {
	}

	inline float Vector2::Magnitude() const {
		return sqrtf(X * X + Y * Y);
	}

	inline float Vector2::MagnitudeSquared() const {
		return X * X + Y * Y;
	}

	inline void Vector2::Normalize() {
		*this /= Magnitude();
	}

	inline Vector2 Vector2::Normalized() const {
		Vector2 result = Vector2(*this);
		return result /= Magnitude();;
	}

	FORCEINLINE float Vector2::Cross(const Vector2& Other) const {
		return X * Other.Y - Y * Other.X;
	}

	FORCEINLINE float Vector2::Cross(const Vector2 & A, const Vector2 & B) {
		return A.X * B.Y - A.Y * B.X;
	}

	FORCEINLINE float Vector2::Dot(const Vector2& Other) const {
		return X * Other.X + Y * Other.Y;
	}

	FORCEINLINE float Vector2::Dot(const Vector2 & A, const Vector2 & B) {
		return A.X * B.X + A.Y * B.Y;
	}

	inline Vector2 Vector2::Orthogonal(bool Polarity) const {
		return Polarity ? Vector2(-Y, X) : Vector2(Y, -X);
	}

	inline Vector2 Vector2::Orthonormal(bool Polarity) const {
		float Length = Magnitude();
		if (Length == 0)
			return Polarity ? Vector2(0) : Vector2(0);
		return Polarity ? Vector2(-Y / Length, X / Length) : Vector2(Y / Length, -X / Length);
	}

	inline const float * Vector2::PointerToValue() const {
		return &X;
	}

	FORCEINLINE Vector2 Vector2::Lerp(const Vector2 & Start, const Vector2 & End, float t) {
		return Vector2((Start.X * (1.0F - t)) + (End.X * t), (Start.Y * (1.0F - t)) + (End.Y * t));
	}

	inline float & Vector2::operator[](unsigned char i) {
		ES_CORE_ASSERT(i <= 1, "Vector2 index out of bounds");
		return ((float*)this)[i];
	}

	inline float const & Vector2::operator[](unsigned char i) const {
		ES_CORE_ASSERT(i <= 1, "Vector2 index out of bounds");
		return ((float*)this)[i];
	}

	FORCEINLINE bool Vector2::operator==(const Vector2& Other) const {
		return (X == Other.X && Y == Other.Y);
	}

	inline HOST_DEVICE bool Vector2::operator!() const {
		return !X || !Y;
	}

	FORCEINLINE bool Vector2::operator!=(const Vector2& Other) const {
		return (X != Other.X || Y != Other.Y);
	}

	FORCEINLINE Vector2 Vector2::operator+(const Vector2& Other) const {
		return Vector2(X + Other.X, Y + Other.Y);
	}

	FORCEINLINE Vector2 Vector2::operator-(const Vector2& Other) const {
		return Vector2(X - Other.X, Y - Other.Y);
	}

	FORCEINLINE Vector2 Vector2::operator-(void) const {
		return Vector2(-X, -Y);
	}

	FORCEINLINE Vector2 Vector2::operator*(const Vector2 & Other) const {
		return Vector2(X * Other.X, Y * Other.Y);
	}

	FORCEINLINE Vector2 Vector2::operator/(const Vector2 & Other) const {
		return Vector2(X / Other.X, Y / Other.Y);
	}

	FORCEINLINE Vector2 Vector2::operator*(const float& Value) const {
		return Vector2(X * Value, Y * Value);
	}

	FORCEINLINE Vector2 Vector2::operator/(const float& Value) const {
		if (Value == 0) return Vector2();
		return Vector2(X / Value, Y / Value);
	}

	FORCEINLINE Vector2& Vector2::operator+=(const Vector2& Other) {
		X += Other.X; Y += Other.Y; return *this;
	}

	FORCEINLINE Vector2& Vector2::operator-=(const Vector2& Other) {
		X -= Other.X; Y -= Other.Y; return *this;
	}

	FORCEINLINE Vector2 & Vector2::operator*=(const Vector2 & Other) {
		X *= Other.X; Y *= Other.Y; return *this;
	}

	FORCEINLINE Vector2 & Vector2::operator/=(const Vector2 & Other) {
		X /= Other.X; Y /= Other.Y; return *this;
	}

	FORCEINLINE Vector2& Vector2::operator*=(const float& Value) {
		X *= Value;	Y *= Value; return *this;
	}

	FORCEINLINE Vector2& Vector2::operator/=(const float& Value) {
		if (Value == 0) X = Y = 0;
		X /= Value; Y /= Value; return *this;
	}

	inline Vector2 operator*(float Value, const Vector2 & Vector) {
		return Vector2(Value * Vector.X, Value * Vector.Y);
	}

	inline Vector2 operator/(float Value, const Vector2 & Vector) {
		return Vector2(Value / Vector.X, Value / Vector.Y);
	}

}