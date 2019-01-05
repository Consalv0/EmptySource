#pragma once

#include <math.h>
#include <stdexcept>

#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"

FORCEINLINE IntVector3::IntVector3()
	: x(0), y(0), z(0) {
}

FORCEINLINE IntVector3::IntVector3(const Vector2& Vector)
	: x((int)Vector.x), y((int)Vector.y), z(0) {
}

FORCEINLINE IntVector3::IntVector3(const Vector3 & Vector)
	: x((int)Vector.x), y((int)Vector.y), z((int)Vector.z) {
}

FORCEINLINE IntVector3::IntVector3(const IntVector3& Vector)
	: x(Vector.x), y(Vector.y), z(Vector.z) {
}

FORCEINLINE IntVector3::IntVector3(const Vector4 & Vector)
	: x((int)Vector.x), y((int)Vector.y), z((int)Vector.z) {
}

FORCEINLINE IntVector3::IntVector3(const int& x, const int& y, const int& z)
	: x(x), y(y), z(z) {
}

FORCEINLINE IntVector3::IntVector3(const int& x, const int& y)
	: x(x), y(y), z(0) {
}

FORCEINLINE IntVector3::IntVector3(const int& Value)
	: x(Value), y(Value), z(Value) {
}

inline float IntVector3::Magnitude() const {
	return sqrtf(x * float(x) + y * float(y) + z * float(z));
}

inline int IntVector3::MagnitudeSquared() const {
	return x * x + y * y + z * z;
}

FORCEINLINE IntVector3 IntVector3::Cross(const IntVector3& Other) const {
	return IntVector3(
		y * Other.z - z * Other.y,
		z * Other.x - x * Other.z,
		x * Other.y - y * Other.x
	);
}

FORCEINLINE int IntVector3::Dot(const IntVector3& Other) const {
	return (x * Other.x) + (y * Other.y) + (z * Other.z);
}

inline Vector3 IntVector3::FloatVector3() const {
	return Vector3(float(x), float(y), float(z));
}

inline const int * IntVector3::PointerToValue() const {
	return &x;
}

inline int & IntVector3::operator[](unsigned int i) {
	switch (i) {
	case 0: return x;
	case 1: return y;
	case 2: return z;
	default: return z;
	}
}

inline int const & IntVector3::operator[](unsigned int i) const {
	switch (i) {
	case 0: return x;
	case 1: return y;
	case 2: return z;
	default: return z;
	}
}

FORCEINLINE IntVector3 IntVector3::operator * (const IntVector3& Other) const {
	return IntVector3(
		x * Other.x,
		y * Other.y,
		z * Other.z
	);
}

FORCEINLINE IntVector3 IntVector3::operator/(const IntVector3 & Other) const {
	return IntVector3(
		x / Other.x,
		y / Other.y,
		z / Other.z
	);
}

FORCEINLINE bool IntVector3::operator==(const IntVector3& Other) {
	return (x == Other.x && y == Other.y && z == Other.z);
}

FORCEINLINE bool IntVector3::operator!=(const IntVector3& Other) {
	return (x != Other.x || y != Other.y || z != Other.z);
}

FORCEINLINE IntVector3 IntVector3::operator+(const IntVector3& Other) const {
	return IntVector3(x + Other.x, y + Other.y, z + Other.z);
}

FORCEINLINE IntVector3 IntVector3::operator-(const IntVector3& Other) const {
	return IntVector3(x - Other.x, y - Other.y, z - Other.z);
}

FORCEINLINE IntVector3 IntVector3::operator-(void) const {
	return IntVector3(-x, -y, -z);
}

FORCEINLINE IntVector3 IntVector3::operator*(const int& Value) const {
	return IntVector3(x * Value, y * Value, z * Value);
}

FORCEINLINE IntVector3 IntVector3::operator/(const int& Value) const {
	if (Value == 0) throw std::exception("Can't divide by zero");
	return IntVector3(x / Value, y / Value, z / Value);
}

FORCEINLINE IntVector3& IntVector3::operator+=(const IntVector3& Other) {
	x += Other.x;
	y += Other.y;
	z += Other.z;
	return *this;
}

FORCEINLINE IntVector3& IntVector3::operator-=(const IntVector3& Other) {
	x -= Other.x;
	y -= Other.y;
	z -= Other.z;
	return *this;
}

FORCEINLINE IntVector3 & IntVector3::operator*=(const IntVector3 & Other) {
	x *= Other.x;
	y *= Other.y;
	z *= Other.z;
	return *this;
}

FORCEINLINE IntVector3 & IntVector3::operator/=(const IntVector3 & Other) {
	x /= Other.x;
	y /= Other.y;
	z /= Other.z;
	return *this;
}

FORCEINLINE IntVector3& IntVector3::operator*=(const int& Value) {
	x *= Value;
	y *= Value;
	z *= Value;
	return *this;
}

FORCEINLINE IntVector3& IntVector3::operator/=(const int& Value) {
	if (Value == 0) throw std::exception("Can't divide by zero");
	x /= Value;
	y /= Value;
	z /= Value;
	return *this;
}

inline WString IntVector3::ToString() {
	return TextFormat(WString(L"{%d, %d, %d}"), x, y, z);
}
