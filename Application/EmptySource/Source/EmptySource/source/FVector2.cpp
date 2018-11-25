#pragma once

#include <math.h>
#include <stdexcept>
#include "..\include\FVector2.h"
#include "..\include\FVector3.h"
#include "..\include\FVector4.h"

FVector2::FVector2()
	: x(0), y(0) {
}

FVector2::FVector2(const FVector2& Vector)
	: x(Vector.x), y(Vector.y) {
}

FVector2::FVector2(const FVector3 & Vector)
	: x(Vector.x), y(Vector.y) {
}

FVector2::FVector2(const FVector4 & Vector)
	: x(Vector.x), y(Vector.y) {
}

FVector2::FVector2(const float& x, const float& y)
	: x(x), y(y) {
}

FVector2::FVector2(const float& Value)
	: x(Value), y(Value) {
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

float FVector2::Cross(const FVector2& Other) const {
	return x * Other.y - y * Other.x;
}

float FVector2::Dot(const FVector2& Other) const {
	return x * Other.x + y * Other.y;
}

const float * FVector2::PoiterToValue() const {
	return &x;
}

FVector2 FVector2::Lerp(const FVector2 & Start, const FVector2 & End, float t) {
	return FVector2((Start.x * (1.0F - t)) + (End.x * t), (Start.y * (1.0F - t)) + (End.y * t));
}

bool FVector2::operator==(const FVector2& Other) {
	return (x == Other.x && y == Other.y);
}

bool FVector2::operator!=(const FVector2& Other) {
	return (x != Other.x || y != Other.y);
}

FVector2 FVector2::operator+(const FVector2& Other) const {
	return FVector2(x + Other.x, y + Other.y);
}

FVector2 FVector2::operator-(const FVector2& Other) const {
	return FVector2(x - Other.x, y - Other.y);
}

FVector2 FVector2::operator-(void) const {
	return FVector2(-x, -y);
}

FVector2 FVector2::operator*(const FVector2 & Other) const {
	return FVector2(x * Other.x, y * Other.y);
}

FVector2 FVector2::operator/(const FVector2 & Other) const {
	return FVector2(x / Other.x, y / Other.y);
}

FVector2 FVector2::operator*(const float& Value) const {
	return FVector2(x * Value, y * Value);
}

FVector2 FVector2::operator/(const float& Value) const {
	if (Value == 0) throw std::exception("Can't divide by zero");
	return FVector2(x / Value, y / Value);
}

FVector2& FVector2::operator+=(const FVector2& Other) {
	x += Other.x;
	y += Other.y;
	return *this;
}

FVector2& FVector2::operator-=(const FVector2& Other) {
	x -= Other.x;
	y -= Other.y;
	return *this;
}

FVector2 & FVector2::operator*=(const FVector2 & Other) {
	x *= Other.x;
	y *= Other.y;
	return *this;
}

FVector2 & FVector2::operator/=(const FVector2 & Other) {
	x /= Other.x;
	y /= Other.y;
	return *this;
}

FVector2& FVector2::operator*=(const float& Value) {
	x *= Value;
	y *= Value;
	return *this;
}

FVector2& FVector2::operator/=(const float& Value) {
	if (Value == 0) throw std::exception("Can't divide by zero");
	x /= Value;
	y /= Value;
	return *this;
}