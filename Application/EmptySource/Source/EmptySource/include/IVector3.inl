#pragma once

#include <math.h>
#include <stdexcept>
#include "..\include\IVector3.h"
#include "..\include\FVector2.h"
#include "..\include\FVector3.h"
#include "..\include\FVector4.h"

FORCEINLINE IVector3::IVector3()
	: x(0), y(0), z(0) {
}

FORCEINLINE IVector3::IVector3(const FVector2& Vector)
	: x((int)Vector.x), y((int)Vector.y), z(0) {
}

FORCEINLINE IVector3::IVector3(const FVector3 & Vector)
	: x((int)Vector.x), y((int)Vector.y), z((int)Vector.z) {
}

FORCEINLINE IVector3::IVector3(const IVector3& Vector)
	: x(Vector.x), y(Vector.y), z(Vector.z) {
}

FORCEINLINE IVector3::IVector3(const FVector4 & Vector)
	: x((int)Vector.x), y((int)Vector.y), z((int)Vector.z) {
}

FORCEINLINE IVector3::IVector3(const int& x, const int& y, const int& z)
	: x(x), y(y), z(z) {
}

FORCEINLINE IVector3::IVector3(const int& x, const int& y)
	: x(x), y(y), z(0) {
}

FORCEINLINE IVector3::IVector3(const int& Value)
	: x(Value), y(Value), z(Value) {
}

inline float IVector3::Magnitude() const {
	return sqrtf(x * float(x) + y * float(y) + z * float(z));
}

inline int IVector3::MagnitudeSquared() const {
	return x * x + y * y + z * z;
}

FORCEINLINE IVector3 IVector3::Cross(const IVector3& Other) const {
	return IVector3(
		y * Other.z - z * Other.y,
		z * Other.x - x * Other.z,
		x * Other.y - y * Other.x
	);
}

FORCEINLINE int IVector3::Dot(const IVector3& Other) const {
	return (x * Other.x) + (y * Other.y) + (z * Other.z);
}

inline FVector3 IVector3::FloatVector3() const {
	return FVector3(float(x), float(y), float(z));
}

inline const int * IVector3::PointerToValue() const {
	return &x;
}

inline int & IVector3::operator[](unsigned int i) {
	switch (i) {
	case 0: return x;
	case 1: return y;
	case 2: return z;
	default: return z;
	}
}

inline int const & IVector3::operator[](unsigned int i) const {
	switch (i) {
	case 0: return x;
	case 1: return y;
	case 2: return z;
	default: return z;
	}
}

FORCEINLINE IVector3 IVector3::operator * (const IVector3& Other) const {
	return IVector3(
		x * Other.x,
		y * Other.y,
		z * Other.z
	);
}

FORCEINLINE IVector3 IVector3::operator/(const IVector3 & Other) const {
	return IVector3(
		x / Other.x,
		y / Other.y,
		z / Other.z
	);
}

FORCEINLINE bool IVector3::operator==(const IVector3& Other) {
	return (x == Other.x && y == Other.y && z == Other.z);
}

FORCEINLINE bool IVector3::operator!=(const IVector3& Other) {
	return (x != Other.x || y != Other.y || z != Other.z);
}

FORCEINLINE IVector3 IVector3::operator+(const IVector3& Other) const {
	return IVector3(x + Other.x, y + Other.y, z + Other.z);
}

FORCEINLINE IVector3 IVector3::operator-(const IVector3& Other) const {
	return IVector3(x - Other.x, y - Other.y, z - Other.z);
}

FORCEINLINE IVector3 IVector3::operator-(void) const {
	return IVector3(-x, -y, -z);
}

FORCEINLINE IVector3 IVector3::operator*(const int& Value) const {
	return IVector3(x * Value, y * Value, z * Value);
}

FORCEINLINE IVector3 IVector3::operator/(const int& Value) const {
	if (Value == 0) throw std::exception("Can't divide by zero");
	return IVector3(x / Value, y / Value, z / Value);
}

FORCEINLINE IVector3& IVector3::operator+=(const IVector3& Other) {
	x += Other.x;
	y += Other.y;
	z += Other.z;
	return *this;
}

FORCEINLINE IVector3& IVector3::operator-=(const IVector3& Other) {
	x -= Other.x;
	y -= Other.y;
	z -= Other.z;
	return *this;
}

FORCEINLINE IVector3 & IVector3::operator*=(const IVector3 & Other) {
	x *= Other.x;
	y *= Other.y;
	z *= Other.z;
	return *this;
}

FORCEINLINE IVector3 & IVector3::operator/=(const IVector3 & Other) {
	x /= Other.x;
	y /= Other.y;
	z /= Other.z;
	return *this;
}

FORCEINLINE IVector3& IVector3::operator*=(const int& Value) {
	x *= Value;
	y *= Value;
	z *= Value;
	return *this;
}

FORCEINLINE IVector3& IVector3::operator/=(const int& Value) {
	if (Value == 0) throw std::exception("Can't divide by zero");
	x /= Value;
	y /= Value;
	z /= Value;
	return *this;
}
