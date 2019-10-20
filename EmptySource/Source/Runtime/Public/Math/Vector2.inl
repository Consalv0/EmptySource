#pragma once

#include <math.h>
#include <stdexcept>

#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/Vector2.h"

namespace ESource {

	FORCEINLINE Vector2::Vector2()
		: x(0), y(0) {
	}

	FORCEINLINE Vector2::Vector2(const Vector2 & Vector)
		: x(Vector.x), y(Vector.y) {
	}

	FORCEINLINE Vector2::Vector2(const Vector3 & Vector)
		: x(Vector.x), y(Vector.y) {
	}

	FORCEINLINE Vector2::Vector2(const Vector4 & Vector)
		: x(Vector.x), y(Vector.y) {
	}

	FORCEINLINE Vector2::Vector2(const float& x, const float& y)
		: x(x), y(y) {
	}

	FORCEINLINE Vector2::Vector2(const float& Value)
		: x(Value), y(Value) {
	}

	inline float Vector2::Magnitude() const {
		return sqrtf(x * x + y * y);
	}

	inline float Vector2::MagnitudeSquared() const {
		return x * x + y * y;
	}

	inline void Vector2::Normalize() {
		*this /= Magnitude();
	}

	inline Vector2 Vector2::Normalized() const {
		Vector2 result = Vector2(*this);
		return result /= Magnitude();;
	}

	FORCEINLINE float Vector2::Cross(const Vector2& Other) const {
		return x * Other.y - y * Other.x;
	}

	FORCEINLINE float Vector2::Cross(const Vector2 & A, const Vector2 & B) {
		return A.x * B.y - A.y * B.x;
	}

	FORCEINLINE float Vector2::Dot(const Vector2& Other) const {
		return x * Other.x + y * Other.y;
	}

	FORCEINLINE float Vector2::Dot(const Vector2 & A, const Vector2 & B) {
		return A.x * B.x + A.y * B.y;
	}

	inline Vector2 Vector2::Orthogonal(bool Polarity) const {
		return Polarity ? Vector2(-y, x) : Vector2(y, -x);
	}

	inline Vector2 Vector2::Orthonormal(bool Polarity) const {
		float Length = Magnitude();
		if (Length == 0)
			return Polarity ? Vector2(0) : Vector2(0);
		return Polarity ? Vector2(-y / Length, x / Length) : Vector2(y / Length, -x / Length);
	}

	inline const float * Vector2::PointerToValue() const {
		return &x;
	}

	FORCEINLINE Vector2 Vector2::Lerp(const Vector2 & Start, const Vector2 & End, float t) {
		return Vector2((Start.x * (1.0F - t)) + (End.x * t), (Start.y * (1.0F - t)) + (End.y * t));
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
		return (x == Other.x && y == Other.y);
	}

	inline HOST_DEVICE bool Vector2::operator!() const {
		return !x || !y;
	}

	FORCEINLINE bool Vector2::operator!=(const Vector2& Other) const {
		return (x != Other.x || y != Other.y);
	}

	FORCEINLINE Vector2 Vector2::operator+(const Vector2& Other) const {
		return Vector2(x + Other.x, y + Other.y);
	}

	FORCEINLINE Vector2 Vector2::operator-(const Vector2& Other) const {
		return Vector2(x - Other.x, y - Other.y);
	}

	FORCEINLINE Vector2 Vector2::operator-(void) const {
		return Vector2(-x, -y);
	}

	FORCEINLINE Vector2 Vector2::operator*(const Vector2 & Other) const {
		return Vector2(x * Other.x, y * Other.y);
	}

	FORCEINLINE Vector2 Vector2::operator/(const Vector2 & Other) const {
		return Vector2(x / Other.x, y / Other.y);
	}

	FORCEINLINE Vector2 Vector2::operator*(const float& Value) const {
		return Vector2(x * Value, y * Value);
	}

	FORCEINLINE Vector2 Vector2::operator/(const float& Value) const {
		if (Value == 0) return Vector2();
		return Vector2(x / Value, y / Value);
	}

	FORCEINLINE Vector2& Vector2::operator+=(const Vector2& Other) {
		x += Other.x; y += Other.y; return *this;
	}

	FORCEINLINE Vector2& Vector2::operator-=(const Vector2& Other) {
		x -= Other.x; y -= Other.y; return *this;
	}

	FORCEINLINE Vector2 & Vector2::operator*=(const Vector2 & Other) {
		x *= Other.x; y *= Other.y; return *this;
	}

	FORCEINLINE Vector2 & Vector2::operator/=(const Vector2 & Other) {
		x /= Other.x; y /= Other.y; return *this;
	}

	FORCEINLINE Vector2& Vector2::operator*=(const float& Value) {
		x *= Value;	y *= Value; return *this;
	}

	FORCEINLINE Vector2& Vector2::operator/=(const float& Value) {
		if (Value == 0) x = y = 0;
		x /= Value; y /= Value; return *this;
	}

	inline Vector2 operator*(float Value, const Vector2 & Vector) {
		return Vector2(Value * Vector.x, Value * Vector.y);
	}

	inline Vector2 operator/(float Value, const Vector2 & Vector) {
		return Vector2(Value / Vector.x, Value / Vector.y);
	}

}