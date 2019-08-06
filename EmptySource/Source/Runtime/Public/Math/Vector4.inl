#pragma once

#include <cmath>
#include <stdexcept>

#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/IntVector3.h"
#include "Math/Vector4.h"

namespace EmptySource {

	FORCEINLINE Vector4::Vector4()
		: x(0), y(0), z(0), w(0) {
	}

	FORCEINLINE Vector4::Vector4(const Vector4& Vector)
		: x(Vector.x), y(Vector.y), z(Vector.z), w(Vector.w) {
	};

	FORCEINLINE Vector4::Vector4(const float& x, const float& y, const float& z)
		: x(x), y(y), z(z), w(0) {
	}

	FORCEINLINE Vector4::Vector4(const float& x, const float& y, const float& z, const float& w)
		: x(x), y(y), z(z), w(w) {
	}

	FORCEINLINE Vector4::Vector4(const Vector2 & Vector)
		: x(Vector.x), y(Vector.y), z(0), w(0) {
	};

	FORCEINLINE Vector4::Vector4(const Vector3 & Vector)
		: x(Vector.x), y(Vector.y), z(Vector.z), w(0) {
	};

	FORCEINLINE Vector4::Vector4(const Vector3 & Vector, const float & w)
		: x(Vector.x), y(Vector.y), z(Vector.z), w(w) {
	};

	FORCEINLINE Vector4::Vector4(const float& Value)
		: x(Value), y(Value), z(Value), w(Value) {
	}

	inline float Vector4::Magnitude() const {
		return sqrtf(x * x + y * y + z * z + w * w);
	}

	inline float Vector4::MagnitudeSquared() const {
		return x * x + y * y + z * z + w * w;
	}

	inline void Vector4::Normalize() {
		if (MagnitudeSquared() == 0) {
			x = 0; y = 0; z = 0; w = 0;
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
		return x * Other.x + y * Other.y + z * Other.z + w * Other.w;
	}

	FORCEINLINE float Vector4::Dot(const Vector4 & A, const Vector4 & B) {
		return A.x * B.x + A.y * B.y + A.z * B.z + A.w * B.w;
	}

	inline const float * Vector4::PointerToValue() const {
		return &x;
	}

	FORCEINLINE Vector4 Vector4::Lerp(const Vector4 & Start, const Vector4 & End, float t) {
		return Vector4(
			(Start.x * (1.0F - t)) + (End.x * t),
			(Start.y * (1.0F - t)) + (End.y * t),
			(Start.z * (1.0F - t)) + (End.z * t),
			(Start.w * (1.0F - t)) + (End.w * t)
		);
	}

	inline float & Vector4::operator[](unsigned int i) {
		if ((i >= 4)) return w;
		return ((float*)this)[i];
	}

	inline float const & Vector4::operator[](unsigned int i) const {
		if ((i >= 4)) return w;
		return ((float*)this)[i];
	}

	FORCEINLINE bool Vector4::operator==(const Vector4& Other) const {
		return (x == Other.x && y == Other.y && z == Other.z && w == Other.w);
	}

	FORCEINLINE bool Vector4::operator!=(const Vector4& Other) const {
		return (x != Other.x || y != Other.y || z != Other.z || w != Other.w);
	}

	FORCEINLINE Vector4 Vector4::operator+(const Vector4& Other) const {
		return Vector4(x + Other.x, y + Other.y, z + Other.z, w + Other.w);
	}

	FORCEINLINE Vector4 Vector4::operator-(const Vector4& Other) const {
		return Vector4(x - Other.x, y - Other.y, z - Other.z, w - Other.w);
	}

	FORCEINLINE Vector4 Vector4::operator-(void) const {
		return Vector4(-x, -y, -z, -w);
	}

	FORCEINLINE Vector4 Vector4::operator*(const float& Value) const {
		return Vector4(x * Value, y * Value, z * Value, w * Value);
	}

	FORCEINLINE Vector4 Vector4::operator/(const float& Value) const {
		if (Value == 0) return Vector4();
		return Vector4(x / Value, y / Value, z / Value, w / Value);
	}

	FORCEINLINE Vector4 Vector4::operator*(const Vector4 & Other) const {
		return Vector4(x * Other.x, y * Other.y, z * Other.z, w * Other.w);
	}

	FORCEINLINE Vector4 Vector4::operator/(const Vector4 & Other) const {
		return Vector4(x / Other.x, y / Other.y, z / Other.z, w / Other.w);
	}

	FORCEINLINE Vector4& Vector4::operator+=(const Vector4& Other) {
		x += Other.x;
		y += Other.y;
		z += Other.z;
		w += Other.w;
		return *this;
	}

	FORCEINLINE Vector4& Vector4::operator-=(const Vector4& Other) {
		x -= Other.x;
		y -= Other.y;
		z -= Other.z;
		w -= Other.w;
		return *this;
	}

	FORCEINLINE Vector4 & Vector4::operator*=(const Vector4 & Other) {
		x *= Other.x;
		y *= Other.y;
		z *= Other.z;
		w *= Other.w;
		return *this;
	}

	FORCEINLINE Vector4 & Vector4::operator/=(const Vector4 & Other) {
		x /= Other.x;
		y /= Other.y;
		z /= Other.z;
		w /= Other.w;
		return *this;
	}

	FORCEINLINE Vector4& Vector4::operator*=(const float& Value) {
		x *= Value;
		y *= Value;
		z *= Value;
		w *= Value;
		return *this;
	}

	FORCEINLINE Vector4& Vector4::operator/=(const float& Value) {
		if (Value == 0) x = y = z = w = 0;
		x /= Value;
		y /= Value;
		z /= Value;
		w /= Value;
		return *this;
	}

	inline Vector4 operator*(float Value, const Vector4 & Vector) {
		return Vector4(Value * Vector.x, Value * Vector.y, Value * Vector.z, Value * Vector.w);
	}

	inline Vector4 operator/(float Value, const Vector4 & Vector) {
		return Vector4(Value / Vector.x, Value / Vector.y, Value / Vector.z, Value / Vector.w);
	}

}