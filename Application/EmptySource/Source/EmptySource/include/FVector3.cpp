#pragma once

#include <math.h>
#include <stdexcept>
#include "FVector2.h"
#include "FVector3.h"

FVector3::FVector3()
	: x(0), y(0), z(0) {
}

FVector3::FVector3(const FVector3& vector)
	: x(vector.x), y(vector.y), z(vector.z) {
};

FVector3::FVector3(const float& x, const float& y, const float& z)
	: x(x), y(y), z(z) {
}

FVector3::FVector3(const FVector2& vector)
	: x(vector.x), y(vector.y), z(0) {
};

FVector3::FVector3(const float& x, const float& y)
	: x(x), y(y), z(0) {
}

FVector3::FVector3(const float& value)
	: x(value), y(value), z(value) {
}

float FVector3::Magnitude() const {
	return sqrtf(x * x + y * y + z * z);
}

float FVector3::MagnitudeSquared() const {
	return x * x + y * y + z * z;
}

void FVector3::Normalize() {
	if (MagnitudeSquared() == 0) {
		x = 0; y = 0; z = 0;
	} else {
		*this /= Magnitude();
	}
}

FVector3 FVector3::Normalized() const {
	if (MagnitudeSquared() == 0) return FVector3();
	return *this / Magnitude();;
}

FVector3 FVector3::Cross(const FVector3& other) const {
	return FVector3(
		y * other.z - z * other.y,
		z * other.x - x * other.z,
		x * other.y - y * other.x
	);
}

float FVector3::Dot(const FVector3& other) const {
	return (x * other.x) + (y * other.y) + (z * other.z);
}

FVector3 FVector3::Lerp(const FVector3 & start, const FVector3 & end, float t) {
	return FVector3((start.x * (1.0F - t)) + (end.x * t), (start.y * (1.0F - t)) + (end.y * t), (start.z * (1.0F - t)) + (end.z * t));
}

FVector3 FVector3::operator * (const FVector3& other) const {
	return FVector3(
		x * other.x,
		y * other.y,
		z * other.z
	);
}

bool FVector3::operator==(const FVector3& other) {
	return (x == other.x && y == other.y && z == other.z);
}

bool FVector3::operator!=(const FVector3& other) {
	return (x != other.x || y != other.y || z != other.z);
}

FVector3 FVector3::operator+(const FVector2& other) const {
	return FVector3(x + other.x, y + other.y, z);
}

FVector3 FVector3::operator-(const FVector2& other) const {
	return FVector3(x - other.x, y - other.y, z);
}

FVector3 FVector3::operator+(const FVector3& other) const {
	return FVector3(x + other.x, y + other.y, z + other.z);
}

FVector3 FVector3::operator-(const FVector3& other) const {
	return FVector3(x - other.x, y - other.y, z - other.z);
}

FVector3 FVector3::operator-(void) const {
	return FVector3(-x, -y, -z);
}

FVector3 FVector3::operator*(const float& value) const {
	return FVector3(x * value, y * value, z * value);
}

FVector3 FVector3::operator/(const float& value) const {
	if (value == 0) throw std::exception("Can't divide by zero");
	return FVector3(x / value, y / value, z / value);
}

FVector3& FVector3::operator+=(const FVector2& other) {
	x += other.x;
	y += other.y;
	return *this;
}

FVector3& FVector3::operator-=(const FVector2& other) {
	x -= other.x;
	y -= other.y;
	return *this;
}

FVector3& FVector3::operator+=(const FVector3& other) {
	x += other.x;
	y += other.y;
	z += other.z;
	return *this;
}

FVector3& FVector3::operator-=(const FVector3& other) {
	x -= other.x;
	y -= other.y;
	z -= other.z;
	return *this;
}

FVector3& FVector3::operator*=(const float& value) {
	x *= value;
	y *= value;
	z *= value;
	return *this;
}

FVector3& FVector3::operator/=(const float& value) {
	if (value == 0) throw std::exception("Can't divide by zero");
	x /= value;
	y /= value;
	z /= value;
	return *this;
}
