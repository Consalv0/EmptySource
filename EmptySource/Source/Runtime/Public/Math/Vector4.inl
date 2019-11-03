#pragma once

#include <cmath>
#include <stdexcept>

#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/IntVector3.h"
#include "Math/Vector4.h"

namespace ESource {

	FORCEINLINE Vector4::Vector4()
		: X(0), Y(0), Z(0), W(0) {
	}

	FORCEINLINE Vector4::Vector4(const Vector4& Vector)
		: X(Vector.X), Y(Vector.Y), Z(Vector.Z), W(Vector.W) {
	};

	FORCEINLINE Vector4::Vector4(const float& X, const float& Y, const float& Z)
		: X(X), Y(Y), Z(Z), W(0) {
	}

	FORCEINLINE Vector4::Vector4(const float& X, const float& Y, const float& Z, const float& W)
		: X(X), Y(Y), Z(Z), W(W) {
	}

	FORCEINLINE Vector4::Vector4(const Vector2 & Vector)
		: X(Vector.X), Y(Vector.Y), Z(0), W(0) {
	};

	FORCEINLINE Vector4::Vector4(const Vector3 & Vector)
		: X(Vector.X), Y(Vector.Y), Z(Vector.Z), W(0) {
	};

	FORCEINLINE Vector4::Vector4(const Vector3 & Vector, const float & W)
		: X(Vector.X), Y(Vector.Y), Z(Vector.Z), W(W) {
	};

	FORCEINLINE Vector4::Vector4(const float& Value)
		: X(Value), Y(Value), Z(Value), W(Value) {
	}

	inline float Vector4::Magnitude() const {
		return sqrtf(X * X + Y * Y + Z * Z + W * W);
	}

	inline float Vector4::MagnitudeSquared() const {
		return X * X + Y * Y + Z * Z + W * W;
	}

	inline void Vector4::Normalize() {
		if (MagnitudeSquared() == 0) {
			X = 0; Y = 0; Z = 0; W = 0;
		}
		else {
			*this /= Magnitude();
		}
	}

	inline Vector4 Vector4::Normalized() const {
		if (MagnitudeSquared() == 0) return Vector4();
		Vector4 result = Vector4(*this);
		return result /= Magnitude();
	}

	FORCEINLINE float Vector4::Dot(const Vector4& Other) const {
		return X * Other.X + Y * Other.Y + Z * Other.Z + W * Other.W;
	}

	FORCEINLINE float Vector4::Dot(const Vector4 & A, const Vector4 & B) {
		return A.X * B.X + A.Y * B.Y + A.Z * B.Z + A.W * B.W;
	}

	inline const float * Vector4::PointerToValue() const {
		return &X;
	}

	FORCEINLINE Vector4 Vector4::Lerp(const Vector4 & Start, const Vector4 & End, float t) {
		return Vector4(
			(Start.X * (1.0F - t)) + (End.X * t),
			(Start.Y * (1.0F - t)) + (End.Y * t),
			(Start.Z * (1.0F - t)) + (End.Z * t),
			(Start.W * (1.0F - t)) + (End.W * t)
		);
	}

	inline float & Vector4::operator[](unsigned char i) {
		ES_CORE_ASSERT(i <= 3, "Vector4 index out of bounds");
		return ((float*)this)[i];
	}

	inline float const & Vector4::operator[](unsigned char i) const {
		ES_CORE_ASSERT(i <= 3, "Vector4 index out of bounds");
		return ((float*)this)[i];
	}

	FORCEINLINE bool Vector4::operator==(const Vector4& Other) const {
		return (X == Other.X && Y == Other.Y && Z == Other.Z && W == Other.W);
	}

	FORCEINLINE bool Vector4::operator!=(const Vector4& Other) const {
		return (X != Other.X || Y != Other.Y || Z != Other.Z || W != Other.W);
	}

	FORCEINLINE Vector4 Vector4::operator+(const Vector4& Other) const {
		return Vector4(X + Other.X, Y + Other.Y, Z + Other.Z, W + Other.W);
	}

	FORCEINLINE Vector4 Vector4::operator-(const Vector4& Other) const {
		return Vector4(X - Other.X, Y - Other.Y, Z - Other.Z, W - Other.W);
	}

	FORCEINLINE Vector4 Vector4::operator-(void) const {
		return Vector4(-X, -Y, -Z, -W);
	}

	FORCEINLINE Vector4 Vector4::operator*(const float& Value) const {
		return Vector4(X * Value, Y * Value, Z * Value, W * Value);
	}

	FORCEINLINE Vector4 Vector4::operator/(const float& Value) const {
		if (Value == 0) return Vector4();
		return Vector4(X / Value, Y / Value, Z / Value, W / Value);
	}

	FORCEINLINE Vector4 Vector4::operator*(const Vector4 & Other) const {
		return Vector4(X * Other.X, Y * Other.Y, Z * Other.Z, W * Other.W);
	}

	FORCEINLINE Vector4 Vector4::operator/(const Vector4 & Other) const {
		return Vector4(X / Other.X, Y / Other.Y, Z / Other.Z, W / Other.W);
	}

	FORCEINLINE Vector4& Vector4::operator+=(const Vector4& Other) {
		X += Other.X;
		Y += Other.Y;
		Z += Other.Z;
		W += Other.W;
		return *this;
	}

	FORCEINLINE Vector4& Vector4::operator-=(const Vector4& Other) {
		X -= Other.X;
		Y -= Other.Y;
		Z -= Other.Z;
		W -= Other.W;
		return *this;
	}

	FORCEINLINE Vector4 & Vector4::operator*=(const Vector4 & Other) {
		X *= Other.X;
		Y *= Other.Y;
		Z *= Other.Z;
		W *= Other.W;
		return *this;
	}

	FORCEINLINE Vector4 & Vector4::operator/=(const Vector4 & Other) {
		X /= Other.X;
		Y /= Other.Y;
		Z /= Other.Z;
		W /= Other.W;
		return *this;
	}

	FORCEINLINE Vector4& Vector4::operator*=(const float& Value) {
		X *= Value;
		Y *= Value;
		Z *= Value;
		W *= Value;
		return *this;
	}

	FORCEINLINE Vector4& Vector4::operator/=(const float& Value) {
		if (Value == 0) X = Y = Z = W = 0;
		X /= Value;
		Y /= Value;
		Z /= Value;
		W /= Value;
		return *this;
	}

	inline Vector4 operator*(float Value, const Vector4 & Vector) {
		return Vector4(Value * Vector.X, Value * Vector.Y, Value * Vector.Z, Value * Vector.W);
	}

	inline Vector4 operator/(float Value, const Vector4 & Vector) {
		return Vector4(Value / Vector.X, Value / Vector.Y, Value / Vector.Z, Value / Vector.W);
	}

}