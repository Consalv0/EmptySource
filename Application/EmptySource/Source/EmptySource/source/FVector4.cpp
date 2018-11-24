#pragma once

#include <math.h>
#include <stdexcept>
#include "../include/FVector2.h"
#include "../include/FVector3.h"
#include "../include/FVector4.h"

FVector4::FVector4()
	: x(0), y(0), z(0), w(0) {
}

FVector4::FVector4(const FVector4& vector)
	: x(vector.x), y(vector.y), z(vector.z), w(vector.w) {
};

FVector4::FVector4(const float& x, const float& y, const float& z)
	: x(x), y(y), z(z), w(0) {
}

FVector4::FVector4(const float& x, const float& y, const float& z, const float& w)
	: x(x), y(y), z(z), w(w) {
}

FVector4::FVector4(const FVector2& vector)
	: x(vector.x), y(vector.y), z(0), w(0) {
};

FVector4::FVector4(const FVector3& vector)
	: x(vector.x), y(vector.y), z(vector.z), w(0) {
};

FVector4::FVector4(const float& value)
	: x(value), y(value), z(value) {
}

float FVector4::Magnitude() const {
	return sqrtf(x * x + y * y + z * z + w * w);
}

float FVector4::MagnitudeSquared() const {
	return x * x + y * y + z * z + w * w;
}

void FVector4::Normalize() {
	if (MagnitudeSquared() == 0) {
		x = 0, y = 0, z = 0;
	} else {
		*this /= Magnitude();
	}
}

FVector4 FVector4::Normalized() const {
	if (MagnitudeSquared() == 0) return FVector4();
	FVector4 result = FVector4(*this);
	return result /= Magnitude();
}

float FVector4::Dot(const FVector4& other) const {
	return x * other.x + y * other.y + z * other.z + w * other.w;
}

FVector3 FVector4::Vector3() const {
	return FVector3(x, y, z);
}

bool FVector4::operator==(const FVector4& other) {
	return (x == other.x && y == other.y && z == other.z && w == other.w);
}

bool FVector4::operator!=(const FVector4& other) {
	return (x != other.x || y != other.y || z != other.z || w != other.w);
}

FVector4 FVector4::operator+(const FVector3& other) const {
	return FVector4(x + other.x, y + other.y, z + other.z, w);
}

FVector4 FVector4::operator-(const FVector3& other) const {
	return FVector4(x - other.x, y - other.y, z - other.z, w);
}

FVector4 FVector4::operator+(const FVector4& other) const {
	return FVector4(x + other.x, y + other.y, z + other.z, w + other.w);
}

FVector4 FVector4::operator-(const FVector4& other) const {
	return FVector4(x - other.x, y - other.y, z - other.z, w - other.w);
}

FVector4 FVector4::operator-(void) const {
	return FVector4(-x, -y, -z, -w);
}

FVector4 FVector4::operator*(const float& value) const {
	return FVector4(x * value, y * value, z * value, w * value);
}

FVector4 FVector4::operator/(const float& value) const {
	if (value == 0) throw std::exception("Can't divide by zero");
	return FVector4(x / value, y / value, z / value, w / value);
}

FVector4& FVector4::operator+=(const FVector3& other) {
	x += other.x;
	y += other.y;
	z += other.z;
	return *this;
}

FVector4& FVector4::operator-=(const FVector3& other) {
	x -= other.x;
	y -= other.y;
	z -= other.z;
	return *this;
}

FVector4& FVector4::operator+=(const FVector4& other) {
	x += other.x;
	y += other.y;
	z += other.z;
	w += other.w;
	return *this;
}

FVector4& FVector4::operator-=(const FVector4& other) {
	x -= other.x;
	y -= other.y;
	z -= other.z;
	w -= other.w;
	return *this;
}

FVector4& FVector4::operator*=(const float& value) {
	x *= value;
	y *= value;
	z *= value;
	w *= value;
	return *this;
}

FVector4& FVector4::operator/=(const float& value) {
	if (value == 0) throw std::exception("Can't divide by zero");
	x /= value;
	y /= value;
	z /= value;
	w /= value;
	return *this;
}
