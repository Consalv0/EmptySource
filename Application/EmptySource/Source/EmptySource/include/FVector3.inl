#pragma once

#include "..\include\SCoreTypes.h"
#include <math.h>
#include <stdexcept>

#include "..\include\FVector2.h"
#include "..\include\FVector3.h"
#include "..\include\FVector4.h"

FORCEINLINE FVector3::FVector3()
	: x(0), y(0), z(0) {
}

FORCEINLINE FVector3::FVector3(const FVector2 & Vector)
	: x(Vector.x), y(Vector.y), z(0) {
};

FORCEINLINE FVector3::FVector3(const FVector3 & Vector)
	: x(Vector.x), y(Vector.y), z(Vector.z) {
}

FORCEINLINE FVector3::FVector3(const FVector4 & Vector)
	: x(Vector.x), y(Vector.y), z(Vector.z) {
}

FORCEINLINE FVector3::FVector3(const float& x, const float& y, const float& z)
	: x(x), y(y), z(z) {
}

FORCEINLINE FVector3::FVector3(const float& x, const float& y)
	: x(x), y(y), z(0) {
}

FORCEINLINE FVector3::FVector3(const float& Value)
	: x(Value), y(Value), z(Value) {
}

inline float FVector3::Magnitude() const {
	return sqrtf(x * x + y * y + z * z);
}

inline float FVector3::MagnitudeSquared() const {
	return x * x + y * y + z * z;
}

inline void FVector3::Normalize() {
	if (MagnitudeSquared() == 0) {
		x = 0; y = 0; z = 0;
	} else {
		*this /= Magnitude();
	}
}

inline FVector3 FVector3::Normalized() const {
	if (MagnitudeSquared() == 0) return FVector3();
	return *this / Magnitude();
}

FORCEINLINE FVector3 FVector3::Cross(const FVector3& Other) const {
	return FVector3(
		y * Other.z - z * Other.y,
		z * Other.x - x * Other.z,
		x * Other.y - y * Other.x
	);
}

FORCEINLINE float FVector3::Dot(const FVector3& Other) const {
	return (x * Other.x) + (y * Other.y) + (z * Other.z);
}

inline FVector2 FVector3::Vector2() const {
	return FVector2(x, y);
}

inline const float * FVector3::PointerToValue() const {
	return &x;
}

FORCEINLINE FVector3 FVector3::Lerp(const FVector3 & start, const FVector3 & end, float t) {
	return FVector3((start.x * (1.0F - t)) + (end.x * t), (start.y * (1.0F - t)) + (end.y * t), (start.z * (1.0F - t)) + (end.z * t));
}

inline float & FVector3::operator[](unsigned int i) {
	switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		default: return z;
	}
}

inline float const & FVector3::operator[](unsigned int i) const {
	switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		default: return z;
	}
}

FORCEINLINE FVector3 FVector3::operator * (const FVector3& Other) const {
	return FVector3(
		x * Other.x,
		y * Other.y,
		z * Other.z
	);
}

FORCEINLINE FVector3 FVector3::operator/(const FVector3 & Other) const {
	return FVector3(
		x / Other.x,
		y / Other.y,
		z / Other.z
	);
}

FORCEINLINE bool FVector3::operator==(const FVector3& Other) {
	return (x == Other.x && y == Other.y && z == Other.z);
}

FORCEINLINE bool FVector3::operator!=(const FVector3& Other) {
	return (x != Other.x || y != Other.y || z != Other.z);
}

FORCEINLINE FVector3 FVector3::operator+(const FVector3& Other) const {
	return FVector3(x + Other.x, y + Other.y, z + Other.z);
}

FORCEINLINE FVector3 FVector3::operator-(const FVector3& Other) const {
	return FVector3(x - Other.x, y - Other.y, z - Other.z);
}

FORCEINLINE FVector3 FVector3::operator-(void) const {
	return FVector3(-x, -y, -z);
}

FORCEINLINE FVector3 FVector3::operator*(const float& Value) const {
	return FVector3(x * Value, y * Value, z * Value);
}

FORCEINLINE FVector3 FVector3::operator/(const float& Value) const {
	if (Value == 0) throw std::exception("Can't divide by zero");
	return FVector3(x / Value, y / Value, z / Value);
}

FORCEINLINE FVector3& FVector3::operator+=(const FVector3& Other) {
	x += Other.x;
	y += Other.y;
	z += Other.z;
	return *this;
}

FORCEINLINE FVector3& FVector3::operator-=(const FVector3& Other) {
	x -= Other.x;
	y -= Other.y;
	z -= Other.z;
	return *this;
}

FORCEINLINE FVector3 & FVector3::operator*=(const FVector3 & Other) {
	x *= Other.x;
	y *= Other.y;
	z *= Other.z;
	return *this;
}

FORCEINLINE FVector3 & FVector3::operator/=(const FVector3 & Other) {
	x /= Other.x;
	y /= Other.y;
	z /= Other.z;
	return *this;
}

FORCEINLINE FVector3& FVector3::operator*=(const float& Value) {
	x *= Value;
	y *= Value;
	z *= Value;
	return *this;
}

FORCEINLINE FVector3& FVector3::operator/=(const float& Value) {
	if (Value == 0) throw std::exception("Can't divide by zero");
	x /= Value;
	y /= Value;
	z /= Value;
	return *this;
}
