#pragma once

#include "CoreTypes.h"

namespace ESource {

	struct Vector2;
	struct Vector3;
	struct Vector4;
	struct IntVector3;

	struct IntVector2 {
	public:
		union {
			struct { int X, Y; };
			struct { int U, V; };
		};

		HOST_DEVICE FORCEINLINE IntVector2();
		HOST_DEVICE FORCEINLINE IntVector2(const IntVector2& Vector);
		HOST_DEVICE FORCEINLINE IntVector2(const IntVector3& Vector);
		HOST_DEVICE FORCEINLINE IntVector2(const Vector2& Vector);
		HOST_DEVICE FORCEINLINE IntVector2(const Vector3& Vector);
		HOST_DEVICE FORCEINLINE IntVector2(const Vector4& Vector);
		HOST_DEVICE FORCEINLINE IntVector2(const int& Value);
		HOST_DEVICE FORCEINLINE IntVector2(const int& X, const int& Y);

		HOST_DEVICE inline float Magnitude() const;
		HOST_DEVICE inline int MagnitudeSquared() const;

		HOST_DEVICE FORCEINLINE int Dot(const IntVector2& Other) const;

		HOST_DEVICE inline Vector2 FloatVector2() const;
		HOST_DEVICE inline int & operator[](unsigned char i);
		HOST_DEVICE inline int const& operator[](unsigned char i) const;
		HOST_DEVICE inline const int* PointerToValue() const;

		HOST_DEVICE FORCEINLINE bool operator==(const IntVector2& Other);
		HOST_DEVICE FORCEINLINE bool operator!=(const IntVector2& Other);

		HOST_DEVICE FORCEINLINE IntVector2 operator+(const IntVector2& Other) const;
		HOST_DEVICE FORCEINLINE IntVector2 operator-(const IntVector2& Other) const;
		HOST_DEVICE FORCEINLINE IntVector2 operator-(void) const;
		HOST_DEVICE FORCEINLINE IntVector2 operator*(const IntVector2& Other) const;
		HOST_DEVICE FORCEINLINE IntVector2 operator/(const IntVector2& Other) const;
		HOST_DEVICE FORCEINLINE IntVector2 operator*(const int& Value) const;
		HOST_DEVICE FORCEINLINE IntVector2 operator/(const int& Value) const;

		HOST_DEVICE FORCEINLINE IntVector2& operator+=(const IntVector2& Other);
		HOST_DEVICE FORCEINLINE IntVector2& operator-=(const IntVector2& Other);
		HOST_DEVICE FORCEINLINE IntVector2& operator*=(const IntVector2& Other);
		HOST_DEVICE FORCEINLINE IntVector2& operator/=(const IntVector2& Other);
		HOST_DEVICE FORCEINLINE IntVector2& operator*=(const int& Value);
		HOST_DEVICE FORCEINLINE IntVector2& operator/=(const int& Value);

		HOST_DEVICE inline friend IntVector2 operator*(int Value, const IntVector2 &Vector);
		HOST_DEVICE inline friend IntVector2 operator/(int Value, const IntVector2 &Vector);
	};

	typedef IntVector2 IntPoint2;
}

#include "Math/IntVector2.inl"