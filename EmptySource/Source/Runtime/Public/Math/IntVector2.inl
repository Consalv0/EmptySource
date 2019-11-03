#pragma once

#include <math.h>
#include <stdexcept>

#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/IntVector2.h"

namespace ESource {

	FORCEINLINE IntVector2::IntVector2()
		: X(0), Y(0) {
	}

	FORCEINLINE IntVector2::IntVector2(const Vector2& Vector)
		: X((int)Vector.X), Y((int)Vector.Y) {
	}

	FORCEINLINE IntVector2::IntVector2(const Vector3 & Vector)
		: X((int)Vector.X), Y((int)Vector.Y) {
	}

	FORCEINLINE IntVector2::IntVector2(const Vector4 & Vector)
		: X((int)Vector.X), Y((int)Vector.Y) {
	}

	FORCEINLINE IntVector2::IntVector2(const IntVector2& Vector)
		: X(Vector.X), Y(Vector.Y) {
	}

	FORCEINLINE IntVector2::IntVector2(const IntVector3 & Vector)
		: X(Vector.X), Y(Vector.Y) {
	}

	FORCEINLINE IntVector2::IntVector2(const int& X, const int& Y)
		: X(X), Y(Y) {
	}

	FORCEINLINE IntVector2::IntVector2(const int& Value)
		: X(Value), Y(Value) {
	}

	inline float IntVector2::Magnitude() const {
		return sqrtf(X * float(X) + Y * float(Y));
	}

	inline int IntVector2::MagnitudeSquared() const {
		return X * X + Y * Y;
	}

	FORCEINLINE int IntVector2::Dot(const IntVector2& Other) const {
		return (X * Other.X) + (Y * Other.Y);
	}

	inline Vector2 IntVector2::FloatVector2() const {
		return Vector2(float(X), float(Y));
	}

	inline const int * IntVector2::PointerToValue() const {
		return &X;
	}

	inline int & IntVector2::operator[](unsigned char i) {
		ES_CORE_ASSERT(i <= 1, "IntVector2 index out of bounds");
		return ((int*)this)[i];
	}

	inline int const & IntVector2::operator[](unsigned char i) const {
		ES_CORE_ASSERT(i <= 1, "IntVector2 index out of bounds");
		return ((int*)this)[i];
	}

	FORCEINLINE IntVector2 IntVector2::operator * (const IntVector2& Other) const {
		return IntVector2(
			X * Other.X,
			Y * Other.Y
		);
	}

	FORCEINLINE IntVector2 IntVector2::operator/(const IntVector2 & Other) const {
		return IntVector2(
			X / Other.X,
			Y / Other.Y
		);
	}

	FORCEINLINE bool IntVector2::operator==(const IntVector2& Other) {
		return (X == Other.X && Y == Other.Y);
	}

	FORCEINLINE bool IntVector2::operator!=(const IntVector2& Other) {
		return (X != Other.X || Y != Other.Y);
	}

	FORCEINLINE IntVector2 IntVector2::operator+(const IntVector2& Other) const {
		return IntVector2(X + Other.X, Y + Other.Y);
	}

	FORCEINLINE IntVector2 IntVector2::operator-(const IntVector2& Other) const {
		return IntVector2(X - Other.X, Y - Other.Y);
	}

	FORCEINLINE IntVector2 IntVector2::operator-(void) const {
		return IntVector2(-X, -Y);
	}

	FORCEINLINE IntVector2 IntVector2::operator*(const int& Value) const {
		return IntVector2(X * Value, Y * Value);
	}

	FORCEINLINE IntVector2 IntVector2::operator/(const int& Value) const {
		if (Value == 0) IntVector2();
		return IntVector2(X / Value, Y / Value);
	}

	FORCEINLINE IntVector2& IntVector2::operator+=(const IntVector2& Other) {
		X += Other.X;
		Y += Other.Y;
		return *this;
	}

	FORCEINLINE IntVector2& IntVector2::operator-=(const IntVector2& Other) {
		X -= Other.X;
		Y -= Other.Y;
		return *this;
	}

	FORCEINLINE IntVector2 & IntVector2::operator*=(const IntVector2 & Other) {
		X *= Other.X;
		Y *= Other.Y;
		return *this;
	}

	FORCEINLINE IntVector2 & IntVector2::operator/=(const IntVector2 & Other) {
		X /= Other.X;
		Y /= Other.Y;
		return *this;
	}

	FORCEINLINE IntVector2& IntVector2::operator*=(const int& Value) {
		X *= Value;
		Y *= Value;
		return *this;
	}

	FORCEINLINE IntVector2& IntVector2::operator/=(const int& Value) {
		if (Value == 0) X = Y = 0;
		X /= Value;
		Y /= Value;
		return *this;
	}

	inline IntVector2 operator*(int Value, const IntVector2 & Vector) {
		return IntVector2(Value * Vector.X, Value * Vector.Y);
	}

	inline IntVector2 operator/(int Value, const IntVector2 & Vector) {
		return IntVector2(Value / Vector.X, Value / Vector.Y);
	}

}