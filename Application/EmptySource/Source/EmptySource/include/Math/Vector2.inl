#pragma once

#include <math.h>
#include <stdexcept>

#include "Vector3.h"
#include "Vector4.h"
#include "Vector2.h"

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

FORCEINLINE float Vector2::Dot(const Vector2& Other) const {
	return x * Other.x + y * Other.y;
}

inline const float * Vector2::PointerToValue() const {
	return &x;
}

FORCEINLINE Vector2 Vector2::Lerp(const Vector2 & Start, const Vector2 & End, float t) {
	return Vector2((Start.x * (1.0F - t)) + (End.x * t), (Start.y * (1.0F - t)) + (End.y * t));
}

inline float & Vector2::operator[](unsigned int i) {
	switch (i) {
		case 0: return x;
		case 1: return y;
		default: return y;
	}
}

inline float const & Vector2::operator[](unsigned int i) const {
	switch (i) {
		case 0: return x;
		case 1: return y;
		default: return y;
	}
}

FORCEINLINE bool Vector2::operator==(const Vector2& Other) const {
	return (x == Other.x && y == Other.y);
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
	if (Value == 0) throw std::exception("Can't divide by zero");
	return Vector2(x / Value, y / Value);
}

FORCEINLINE Vector2& Vector2::operator+=(const Vector2& Other) {
	x += Other.x;
	y += Other.y;
	return *this;
}

FORCEINLINE Vector2& Vector2::operator-=(const Vector2& Other) {
	x -= Other.x;
	y -= Other.y;
	return *this;
}

FORCEINLINE Vector2 & Vector2::operator*=(const Vector2 & Other) {
	x *= Other.x;
	y *= Other.y;
	return *this;
}

FORCEINLINE Vector2 & Vector2::operator/=(const Vector2 & Other) {
	x /= Other.x;
	y /= Other.y;
	return *this;
}

FORCEINLINE Vector2& Vector2::operator*=(const float& Value) {
	x *= Value;
	y *= Value;
	return *this;
}

FORCEINLINE Vector2& Vector2::operator/=(const float& Value) {
	if (Value == 0) throw std::exception("Can't divide by zero");
	x /= Value;
	y /= Value;
	return *this;
}

