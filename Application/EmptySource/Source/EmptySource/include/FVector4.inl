#pragma once

#include <math.h>
#include <stdexcept>
#include "../include/FVector2.h"
#include "../include/FVector3.h"
#include "../include/FVector4.h"

FORCEINLINE FVector4::FVector4()
	: x(0), y(0), z(0), w(0) {
}

FORCEINLINE FVector4::FVector4(const FVector4& Vector)
	: x(Vector.x), y(Vector.y), z(Vector.z), w(Vector.w) {
};

FORCEINLINE FVector4::FVector4(const float& x, const float& y, const float& z)
	: x(x), y(y), z(z), w(0) {
}

FORCEINLINE FVector4::FVector4(const float& x, const float& y, const float& z, const float& w)
	: x(x), y(y), z(z), w(w) {
}

FORCEINLINE FVector4::FVector4(const FVector2& Vector)
	: x(Vector.x), y(Vector.y), z(0), w(0) {
};

FORCEINLINE FVector4::FVector4(const FVector3& Vector)
	: x(Vector.x), y(Vector.y), z(Vector.z), w(0) {
};

FORCEINLINE FVector4::FVector4(const float& Value)
	: x(Value), y(Value), z(Value) {
}

inline float FVector4::Magnitude() const {
	return sqrtf(x * x + y * y + z * z + w * w);
}

inline float FVector4::MagnitudeSquared() const {
	return x * x + y * y + z * z + w * w;
}

inline void FVector4::Normalize() {
	if (MagnitudeSquared() == 0) {
		x = 0, y = 0, z = 0;
	} else {
		*this /= Magnitude();
	}
}

inline FVector4 FVector4::Normalized() const {
	if (MagnitudeSquared() == 0) return FVector4();
	FVector4 result = FVector4(*this);
	return result /= Magnitude();
}

FORCEINLINE float FVector4::Dot(const FVector4& Other) const {
	return x * Other.x + y * Other.y + z * Other.z + w * Other.w;
}

inline FVector3 FVector4::Vector3() const {
	return FVector3(*this);
}

inline FVector2 FVector4::Vector2() const {
	return FVector2(*this);
}

inline const float * FVector4::PointerToValue() const {
	return &x;
}

FORCEINLINE FVector4 FVector4::Lerp(const FVector4 & Start, const FVector4 & End, float t) {
	return FVector4(
		(Start.x * (1.0F - t)) + (End.x * t),
		(Start.y * (1.0F - t)) + (End.y * t),
		(Start.z * (1.0F - t)) + (End.z * t),
		(Start.w * (1.0F - t)) + (End.w * t)
	);
}

inline float & FVector4::operator[](unsigned int i) {
	switch (i) {
		case 0:  return x;
		case 1:  return y; 
		case 2:  return z; 
		case 3:  return w; 
		default: return w;
	}
}

inline float const & FVector4::operator[](unsigned int i) const {
	switch (i) {
		case 0:  return x;
		case 1:  return y;
		case 2:  return z;
		case 3:  return w;
		default: return w;
	}
}

FORCEINLINE bool FVector4::operator==(const FVector4& Other) {
	return (x == Other.x && y == Other.y && z == Other.z && w == Other.w);
}

FORCEINLINE bool FVector4::operator!=(const FVector4& Other) {
	return (x != Other.x || y != Other.y || z != Other.z || w != Other.w);
}

FORCEINLINE FVector4 FVector4::operator+(const FVector4& Other) const {
	return FVector4(x + Other.x, y + Other.y, z + Other.z, w + Other.w);
}

FORCEINLINE FVector4 FVector4::operator-(const FVector4& Other) const {
	return FVector4(x - Other.x, y - Other.y, z - Other.z, w - Other.w);
}

FORCEINLINE FVector4 FVector4::operator-(void) const {
	return FVector4(-x, -y, -z, -w);
}

FORCEINLINE FVector4 FVector4::operator*(const float& Value) const {
	return FVector4(x * Value, y * Value, z * Value, w * Value);
}

FORCEINLINE FVector4 FVector4::operator/(const float& Value) const {
	if (Value == 0) throw std::exception("Can't divide by zero");
	return FVector4(x / Value, y / Value, z / Value, w / Value);
}

FORCEINLINE FVector4 FVector4::operator*(const FVector4 & Other) const {
	return FVector4(x * Other.x, y * Other.y, z * Other.z, w * Other.w);
}

FORCEINLINE FVector4 FVector4::operator/(const FVector4 & Other) const {
	return FVector4(x / Other.x, y / Other.y, z / Other.z, w / Other.w);
}

FORCEINLINE FVector4& FVector4::operator+=(const FVector4& Other) {
	x += Other.x;
	y += Other.y;
	z += Other.z;
	w += Other.w;
	return *this;
}

FORCEINLINE FVector4& FVector4::operator-=(const FVector4& Other) {
	x -= Other.x;
	y -= Other.y;
	z -= Other.z;
	w -= Other.w;
	return *this;
}

FORCEINLINE FVector4 & FVector4::operator*=(const FVector4 & Other) {
	x *= Other.x;
	y *= Other.y;
	z *= Other.z;
	w *= Other.w;
	return *this;
}

FORCEINLINE FVector4 & FVector4::operator/=(const FVector4 & Other) {
	x /= Other.x;
	y /= Other.y;
	z /= Other.z;
	w /= Other.w;
	return *this;
}

FORCEINLINE FVector4& FVector4::operator*=(const float& Value) {
	x *= Value;
	y *= Value;
	z *= Value;
	w *= Value;
	return *this;
}

FORCEINLINE FVector4& FVector4::operator/=(const float& Value) {
	if (Value == 0) throw std::exception("Can't divide by zero");
	x /= Value;
	y /= Value;
	z /= Value;
	w /= Value;
	return *this;
}
