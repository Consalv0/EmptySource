#pragma once

#include <math.h>
#include <stdexcept>
#include "..\include\FVector2.h"

FVector2::FVector2()
	: x(0), y(0) {
}

FVector2::FVector2(const FVector2& vector)
	: x(vector.x), y(vector.y) {
}

FVector2::FVector2(const float& x, const float& y)
	: x(x), y(y) {
}

FVector2::FVector2(const float& value)
	: x(value), y(value) {
}

float FVector2::Magnitude() const {
	return sqrtf(x * x + y * y);
}

float FVector2::MagnitudeSquared() const {
	return x * x + y * y;
}

void FVector2::Normalize() {
	*this /= Magnitude();
}

FVector2 FVector2::Normalized() const {
	FVector2 result = FVector2(*this);
	return result /= Magnitude();;
}

float FVector2::Cross(const FVector2& other) const {
	return x * other.y - y * other.x;
}

float FVector2::Dot(const FVector2& other) const {
	return x * other.x + y * other.y;
}

bool FVector2::operator==(const FVector2& other) {
	return (x == other.x && y == other.y);
}

bool FVector2::operator!=(const FVector2& other) {
	return (x != other.x || y != other.y);
}

FVector2 FVector2::operator+(const FVector2& other) const {
	return FVector2(x + other.x, y + other.y);
}

FVector2 FVector2::operator-(const FVector2& other) const {
	return FVector2(x - other.x, y - other.y);
}

FVector2 FVector2::operator-(void) const {
	return FVector2(-x, -y);
}

FVector2 FVector2::operator*(const float& value) const {
	return FVector2(x * value, y * value);
}

FVector2 FVector2::operator/(const float& value) const {
	if (value == 0) throw std::exception("Can't divide by zero");
	return FVector2(x / value, y / value);
}

FVector2& FVector2::operator+=(const FVector2& other) {
	x += other.x;
	y += other.y;
	return *this;
}

FVector2& FVector2::operator-=(const FVector2& other) {
	x -= other.x;
	y -= other.y;
	return *this;
}

FVector2& FVector2::operator*=(const float& value) {
	x *= value;
	y *= value;
	return *this;
}

FVector2& FVector2::operator/=(const float& value) {
	if (value == 0) throw std::exception("Can't divide by zero");
	x /= value;
	y /= value;
	return *this;
}