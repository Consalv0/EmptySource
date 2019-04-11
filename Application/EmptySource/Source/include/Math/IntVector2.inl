#pragma once

#include <math.h>
#include <stdexcept>

#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"
#include "IntVector2.h"

FORCEINLINE IntVector2::IntVector2()
	: x(0), y(0) {
}

FORCEINLINE IntVector2::IntVector2(const Vector2& Vector)
	: x((int)Vector.x), y((int)Vector.y) {
}

FORCEINLINE IntVector2::IntVector2(const Vector3 & Vector)
	: x((int)Vector.x), y((int)Vector.y) {
}

FORCEINLINE IntVector2::IntVector2(const Vector4 & Vector)
	: x((int)Vector.x), y((int)Vector.y) {
}

FORCEINLINE IntVector2::IntVector2(const IntVector2& Vector)
	: x(Vector.x), y(Vector.y) {
}

FORCEINLINE IntVector2::IntVector2(const IntVector3 & Vector)
	: x(Vector.x), y(Vector.y) {
}

FORCEINLINE IntVector2::IntVector2(const int& x, const int& y)
	: x(x), y(y) {
}

FORCEINLINE IntVector2::IntVector2(const int& Value)
	: x(Value), y(Value) {
}

inline float IntVector2::Magnitude() const {
	return sqrtf(x * float(x) + y * float(y));
}

inline int IntVector2::MagnitudeSquared() const {
	return x * x + y * y;
}

FORCEINLINE int IntVector2::Dot(const IntVector2& Other) const {
	return (x * Other.x) + (y * Other.y);
}

inline Vector2 IntVector2::FloatVector2() const {
	return Vector2(float(x), float(y));
}

inline const int * IntVector2::PointerToValue() const {
	return &x;
}

inline int & IntVector2::operator[](unsigned int i) {
	if ((i >= 2)) return y;
	return ((int*)this)[i];
}

inline int const & IntVector2::operator[](unsigned int i) const {
	if ((i >= 2)) return y;
	return ((int*)this)[i];
}

FORCEINLINE IntVector2 IntVector2::operator * (const IntVector2& Other) const {
	return IntVector2(
		x * Other.x,
		y * Other.y
	);
}

FORCEINLINE IntVector2 IntVector2::operator/(const IntVector2 & Other) const {
	return IntVector2(
		x / Other.x,
		y / Other.y
	);
}

FORCEINLINE bool IntVector2::operator==(const IntVector2& Other) {
	return (x == Other.x && y == Other.y);
}

FORCEINLINE bool IntVector2::operator!=(const IntVector2& Other) {
	return (x != Other.x || y != Other.y);
}

FORCEINLINE IntVector2 IntVector2::operator+(const IntVector2& Other) const {
	return IntVector2(x + Other.x, y + Other.y);
}

FORCEINLINE IntVector2 IntVector2::operator-(const IntVector2& Other) const {
	return IntVector2(x - Other.x, y - Other.y);
}

FORCEINLINE IntVector2 IntVector2::operator-(void) const {
	return IntVector2(-x, -y);
}

FORCEINLINE IntVector2 IntVector2::operator*(const int& Value) const {
	return IntVector2(x * Value, y * Value);
}

FORCEINLINE IntVector2 IntVector2::operator/(const int& Value) const {
	if (Value == 0) IntVector2();
	return IntVector2(x / Value, y / Value);
}

FORCEINLINE IntVector2& IntVector2::operator+=(const IntVector2& Other) {
	x += Other.x;
	y += Other.y;
	return *this;
}

FORCEINLINE IntVector2& IntVector2::operator-=(const IntVector2& Other) {
	x -= Other.x;
	y -= Other.y;
	return *this;
}

FORCEINLINE IntVector2 & IntVector2::operator*=(const IntVector2 & Other) {
	x *= Other.x;
	y *= Other.y;
	return *this;
}

FORCEINLINE IntVector2 & IntVector2::operator/=(const IntVector2 & Other) {
	x /= Other.x;
	y /= Other.y;
	return *this;
}

FORCEINLINE IntVector2& IntVector2::operator*=(const int& Value) {
	x *= Value;
	y *= Value;
	return *this;
}

FORCEINLINE IntVector2& IntVector2::operator/=(const int& Value) {
	if (Value == 0) x = y = 0;
	x /= Value;
	y /= Value;
	return *this;
}

inline IntVector2 operator*(int Value, const IntVector2 & Vector) {
	return IntVector2(Value * Vector.x, Value * Vector.y);
}

inline IntVector2 operator/(int Value, const IntVector2 & Vector) {
	return IntVector2(Value / Vector.x, Value / Vector.y);
}