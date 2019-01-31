#pragma once

#include <math.h>
#include <stdexcept>

#include "Vector2.h"
#include "Vector3.h"
#include "IntVector3.h"
#include "Vector4.h"

FORCEINLINE Vector4::Vector4()
	: x(0), y(0), z(0), w(0) {
}

FORCEINLINE Vector4::Vector4(const Vector4& Vector)
	: x(Vector.x), y(Vector.y), z(Vector.z), w(Vector.w) {
};

FORCEINLINE Vector4::Vector4(const float& x, const float& y, const float& z)
	: x(x), y(y), z(z), w(0) {
}

FORCEINLINE Vector4::Vector4(const float& x, const float& y, const float& z, const float& w)
	: x(x), y(y), z(z), w(w) {
}

FORCEINLINE Vector4::Vector4(const Vector2 & Vector)
	: x(Vector.x), y(Vector.y), z(0), w(0) {
};

FORCEINLINE Vector4::Vector4(const Vector3 & Vector)
	: x(Vector.x), y(Vector.y), z(Vector.z), w(0) {
};

FORCEINLINE Vector4::Vector4(const float& Value)
	: x(Value), y(Value), z(Value) {
}

inline float Vector4::Magnitude() const {
	return sqrtf(x * x + y * y + z * z + w * w);
}

inline float Vector4::MagnitudeSquared() const {
	return x * x + y * y + z * z + w * w;
}

inline void Vector4::Normalize() {
	if (MagnitudeSquared() == 0) {
		x = 0, y = 0, z = 0; w = 0;
	} else {
		*this /= Magnitude();
	}
}

inline Vector4 Vector4::Normalized() const {
	if (MagnitudeSquared() == 0) return Vector4();
	Vector4 result = Vector4(*this);
	return result /= Magnitude();
}

FORCEINLINE float Vector4::Dot(const Vector4& Other) const {
	return x * Other.x + y * Other.y + z * Other.z + w * Other.w;
}

inline const float * Vector4::PointerToValue() const {
	return &x;
}

FORCEINLINE Vector4 Vector4::Lerp(const Vector4 & Start, const Vector4 & End, float t) {
	return Vector4(
		(Start.x * (1.0F - t)) + (End.x * t),
		(Start.y * (1.0F - t)) + (End.y * t),
		(Start.z * (1.0F - t)) + (End.z * t),
		(Start.w * (1.0F - t)) + (End.w * t)
	);
}

inline float & Vector4::operator[](unsigned int i) {
	switch (i) {
		case 0:  return x;
		case 1:  return y; 
		case 2:  return z; 
		case 3:  return w; 
		default: return w;
	}
}

inline float const & Vector4::operator[](unsigned int i) const {
	switch (i) {
		case 0:  return x;
		case 1:  return y;
		case 2:  return z;
		case 3:  return w;
		default: return w;
	}
}

FORCEINLINE bool Vector4::operator==(const Vector4& Other) const {
	return (x == Other.x && y == Other.y && z == Other.z && w == Other.w);
}

FORCEINLINE bool Vector4::operator!=(const Vector4& Other) const {
	return (x != Other.x || y != Other.y || z != Other.z || w != Other.w);
}

FORCEINLINE Vector4 Vector4::operator+(const Vector4& Other) const {
	return Vector4(x + Other.x, y + Other.y, z + Other.z, w + Other.w);
}

FORCEINLINE Vector4 Vector4::operator-(const Vector4& Other) const {
	return Vector4(x - Other.x, y - Other.y, z - Other.z, w - Other.w);
}

FORCEINLINE Vector4 Vector4::operator-(void) const {
	return Vector4(-x, -y, -z, -w);
}

FORCEINLINE Vector4 Vector4::operator*(const float& Value) const {
	return Vector4(x * Value, y * Value, z * Value, w * Value);
}

FORCEINLINE Vector4 Vector4::operator/(const float& Value) const {
	if (Value == 0) throw std::exception("Can't divide by zero");
	return Vector4(x / Value, y / Value, z / Value, w / Value);
}

FORCEINLINE Vector4 Vector4::operator*(const Vector4 & Other) const {
	return Vector4(x * Other.x, y * Other.y, z * Other.z, w * Other.w);
}

FORCEINLINE Vector4 Vector4::operator/(const Vector4 & Other) const {
	return Vector4(x / Other.x, y / Other.y, z / Other.z, w / Other.w);
}

FORCEINLINE Vector4& Vector4::operator+=(const Vector4& Other) {
	x += Other.x;
	y += Other.y;
	z += Other.z;
	w += Other.w;
	return *this;
}

FORCEINLINE Vector4& Vector4::operator-=(const Vector4& Other) {
	x -= Other.x;
	y -= Other.y;
	z -= Other.z;
	w -= Other.w;
	return *this;
}

FORCEINLINE Vector4 & Vector4::operator*=(const Vector4 & Other) {
	x *= Other.x;
	y *= Other.y;
	z *= Other.z;
	w *= Other.w;
	return *this;
}

FORCEINLINE Vector4 & Vector4::operator/=(const Vector4 & Other) {
	x /= Other.x;
	y /= Other.y;
	z /= Other.z;
	w /= Other.w;
	return *this;
}

FORCEINLINE Vector4& Vector4::operator*=(const float& Value) {
	x *= Value;
	y *= Value;
	z *= Value;
	w *= Value;
	return *this;
}

FORCEINLINE Vector4& Vector4::operator/=(const float& Value) {
	if (Value == 0) throw std::exception("Can't divide by zero");
	x /= Value;
	y /= Value;
	z /= Value;
	w /= Value;
	return *this;
}

inline WString Vector4::ToString() {
	return Text::Formatted(L"{%.2f, %.2f, %.2f, %.2f}", x, y, z, w);
}
