#pragma once

#include <cmath>
#include <stdexcept>

#include "IntVector2.h"
#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"

namespace ESource {

	FORCEINLINE IntVector3::IntVector3()
		: X(0), Y(0), Z(0) {
	}

	FORCEINLINE IntVector3::IntVector3(const Vector2& Vector)
		: X((int)Vector.X), Y((int)Vector.Y), Z(0) {
	}

	FORCEINLINE IntVector3::IntVector3(const Vector3 & Vector)
		: X((int)Vector.X), Y((int)Vector.Y), Z((int)Vector.Z) {
	}

	FORCEINLINE IntVector3::IntVector3(const IntVector3& Vector)
		: X(Vector.X), Y(Vector.Y), Z(Vector.Z) {
	}

	FORCEINLINE IntVector3::IntVector3(const IntVector2 & Vector)
		: X(Vector.X), Y(Vector.Y), Z(0) {
	}

	FORCEINLINE IntVector3::IntVector3(const Vector4 & Vector)
		: X((int)Vector.X), Y((int)Vector.Y), Z((int)Vector.Z) {
	}

	FORCEINLINE IntVector3::IntVector3(const int& X, const int& Y, const int& Z)
		: X(X), Y(Y), Z(Z) {
	}

	FORCEINLINE IntVector3::IntVector3(const int& X, const int& Y)
		: X(X), Y(Y), Z(0) {
	}

	FORCEINLINE IntVector3::IntVector3(const int& Value)
		: X(Value), Y(Value), Z(Value) {
	}

	inline float IntVector3::Magnitude() const {
		return sqrtf(X * float(X) + Y * float(Y) + Z * float(Z));
	}

	inline int IntVector3::MagnitudeSquared() const {
		return X * X + Y * Y + Z * Z;
	}

	FORCEINLINE IntVector3 IntVector3::Cross(const IntVector3& Other) const {
		return IntVector3(
			Y * Other.Z - Z * Other.Y,
			Z * Other.X - X * Other.Z,
			X * Other.Y - Y * Other.X
		);
	}

	FORCEINLINE int IntVector3::Dot(const IntVector3& Other) const {
		return (X * Other.X) + (Y * Other.Y) + (Z * Other.Z);
	}

	inline Vector3 IntVector3::FloatVector3() const {
		return Vector3(float(X), float(Y), float(Z));
	}

	inline const int * IntVector3::PointerToValue() const {
		return &X;
	}

	inline int & IntVector3::operator[](unsigned char i) {
		ES_CORE_ASSERT(i <= 2, "IntVector3 index out of bounds");
		return ((int*)this)[i];
	}

	inline int const & IntVector3::operator[](unsigned char i) const {
		ES_CORE_ASSERT(i <= 2, "IntVector3 index out of bounds");
		return ((int*)this)[i];
	}

	FORCEINLINE IntVector3 IntVector3::operator * (const IntVector3& Other) const {
		return IntVector3(
			X * Other.X,
			Y * Other.Y,
			Z * Other.Z
		);
	}

	FORCEINLINE IntVector3 IntVector3::operator/(const IntVector3 & Other) const {
		return IntVector3(
			X / Other.X,
			Y / Other.Y,
			Z / Other.Z
		);
	}

	FORCEINLINE bool IntVector3::operator==(const IntVector3& Other) {
		return (X == Other.X && Y == Other.Y && Z == Other.Z);
	}

	FORCEINLINE bool IntVector3::operator!=(const IntVector3& Other) {
		return (X != Other.X || Y != Other.Y || Z != Other.Z);
	}

	FORCEINLINE IntVector3 IntVector3::operator+(const IntVector3& Other) const {
		return IntVector3(X + Other.X, Y + Other.Y, Z + Other.Z);
	}

	FORCEINLINE IntVector3 IntVector3::operator-(const IntVector3& Other) const {
		return IntVector3(X - Other.X, Y - Other.Y, Z - Other.Z);
	}

	FORCEINLINE IntVector3 IntVector3::operator-(void) const {
		return IntVector3(-X, -Y, -Z);
	}

	FORCEINLINE IntVector3 IntVector3::operator*(const int& Value) const {
		return IntVector3(X * Value, Y * Value, Z * Value);
	}

	FORCEINLINE IntVector3 IntVector3::operator/(const int& Value) const {
		if (Value == 0) return IntVector3();
		return IntVector3(X / Value, Y / Value, Z / Value);
	}

	FORCEINLINE IntVector3& IntVector3::operator+=(const IntVector3& Other) {
		X += Other.X;
		Y += Other.Y;
		Z += Other.Z;
		return *this;
	}

	FORCEINLINE IntVector3& IntVector3::operator-=(const IntVector3& Other) {
		X -= Other.X;
		Y -= Other.Y;
		Z -= Other.Z;
		return *this;
	}

	FORCEINLINE IntVector3 & IntVector3::operator*=(const IntVector3 & Other) {
		X *= Other.X;
		Y *= Other.Y;
		Z *= Other.Z;
		return *this;
	}

	FORCEINLINE IntVector3 & IntVector3::operator/=(const IntVector3 & Other) {
		X /= Other.X;
		Y /= Other.Y;
		Z /= Other.Z;
		return *this;
	}

	FORCEINLINE IntVector3& IntVector3::operator*=(const int& Value) {
		X *= Value;
		Y *= Value;
		Z *= Value;
		return *this;
	}

	FORCEINLINE IntVector3& IntVector3::operator/=(const int& Value) {
		if (Value == 0) X = Y = Z = 0;
		X /= Value;
		Y /= Value;
		Z /= Value;
		return *this;
	}

	inline IntVector3 operator*(int Value, const IntVector3 & Vector) {
		return IntVector3(Value * Vector.X, Value * Vector.Y, Value / Vector.Z);
	}

	inline IntVector3 operator/(int Value, const IntVector3 & Vector) {
		return IntVector3(Value / Vector.X, Value / Vector.Y, Value / Vector.Z);
	}

}