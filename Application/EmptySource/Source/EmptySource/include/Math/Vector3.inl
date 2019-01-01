#pragma once

#include <math.h>
#include <stdexcept>
#include <string>

#include "Vector2.h"
#include "IntVector3.h"
#include "Vector4.h"

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
	} else {
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

FORCEINLINE float Vector3::Dot(const Vector3& Other) const {
	return (x * Other.x) + (y * Other.y) + (z * Other.z);
}

inline const float * Vector3::PointerToValue() const {
	return &x;
}

FORCEINLINE Vector3 Vector3::Lerp(const Vector3 & start, const Vector3 & end, float t) {
	return Vector3((start.x * (1.0F - t)) + (end.x * t), (start.y * (1.0F - t)) + (end.y * t), (start.z * (1.0F - t)) + (end.z * t));
}

inline float & Vector3::operator[](unsigned int i) {
	switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		default: return z;
	}
}

inline float const & Vector3::operator[](unsigned int i) const {
	switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		default: return z;
	}
}

FORCEINLINE Vector3 Vector3::operator * (const Vector3& Other) const {
	return Vector3(
		x * Other.x,
		y * Other.y,
		z * Other.z
	);
}

FORCEINLINE Vector3 Vector3::operator/(const Vector3 & Other) const {
	return Vector3(
		x / Other.x,
		y / Other.y,
		z / Other.z
	);
}

FORCEINLINE bool Vector3::operator==(const Vector3& Other) {
	return (x == Other.x && y == Other.y && z == Other.z);
}

FORCEINLINE bool Vector3::operator!=(const Vector3& Other) {
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
	if (Value == 0) throw std::exception("Can't divide by zero");
	return Vector3(x / Value, y / Value, z / Value);
}

FORCEINLINE Vector3& Vector3::operator+=(const Vector3& Other) {
	x += Other.x;
	y += Other.y;
	z += Other.z;
	return *this;
}

FORCEINLINE Vector3& Vector3::operator-=(const Vector3& Other) {
	x -= Other.x;
	y -= Other.y;
	z -= Other.z;
	return *this;
}

FORCEINLINE Vector3 & Vector3::operator*=(const Vector3 & Other) {
	x *= Other.x;
	y *= Other.y;
	z *= Other.z;
	return *this;
}

FORCEINLINE Vector3 & Vector3::operator/=(const Vector3 & Other) {
	x /= Other.x;
	y /= Other.y;
	z /= Other.z;
	return *this;
}

FORCEINLINE Vector3& Vector3::operator*=(const float& Value) {
	x *= Value;
	y *= Value;
	z *= Value;
	return *this;
}

FORCEINLINE Vector3& Vector3::operator/=(const float& Value) {
	if (Value == 0) throw std::exception("Can't divide by zero");
	x /= Value;
	y /= Value;
	z /= Value;
	return *this;
}

inline WString Vector3::ToString() {
	return L"{" + std::to_wstring(x) + L", " + std::to_wstring(y) + L", " + std::to_wstring(z) + L"}";
}
