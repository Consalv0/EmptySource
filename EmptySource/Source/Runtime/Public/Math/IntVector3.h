#pragma once

#include "CoreTypes.h"

namespace EmptySource {

	struct Vector2;
	struct Vector3;
	struct Vector4;

	struct IntVector3 {
	public:
		union {
			struct { int x, y, z; };
			struct { int r, g, b; };
		};

		HOST_DEVICE FORCEINLINE IntVector3();
		HOST_DEVICE FORCEINLINE IntVector3(const IntVector3& Vector);
		HOST_DEVICE FORCEINLINE IntVector3(const Vector2& Vector);
		HOST_DEVICE FORCEINLINE IntVector3(const Vector3& Vector);
		HOST_DEVICE FORCEINLINE IntVector3(const Vector4& Vector);
		HOST_DEVICE FORCEINLINE IntVector3(const int& Value);
		HOST_DEVICE FORCEINLINE IntVector3(const int& x, const int& y, const int& z);
		HOST_DEVICE FORCEINLINE IntVector3(const int& x, const int& y);

		HOST_DEVICE inline float Magnitude() const;
		HOST_DEVICE inline int MagnitudeSquared() const;

		HOST_DEVICE FORCEINLINE IntVector3 Cross(const IntVector3& Other) const;
		HOST_DEVICE FORCEINLINE int Dot(const IntVector3& Other) const;

		HOST_DEVICE inline Vector3 FloatVector3() const;
		HOST_DEVICE inline int & operator[](unsigned int i);
		HOST_DEVICE inline int const& operator[](unsigned int i) const;
		HOST_DEVICE inline const int* PointerToValue() const;

		HOST_DEVICE FORCEINLINE bool operator==(const IntVector3& Other);
		HOST_DEVICE FORCEINLINE bool operator!=(const IntVector3& Other);

		HOST_DEVICE FORCEINLINE IntVector3 operator+(const IntVector3& Other) const;
		HOST_DEVICE FORCEINLINE IntVector3 operator-(const IntVector3& Other) const;
		HOST_DEVICE FORCEINLINE IntVector3 operator-(void) const;
		HOST_DEVICE FORCEINLINE IntVector3 operator*(const IntVector3& Other) const;
		HOST_DEVICE FORCEINLINE IntVector3 operator/(const IntVector3& Other) const;
		HOST_DEVICE FORCEINLINE IntVector3 operator*(const int& Value) const;
		HOST_DEVICE FORCEINLINE IntVector3 operator/(const int& Value) const;

		HOST_DEVICE FORCEINLINE IntVector3& operator+=(const IntVector3& Other);
		HOST_DEVICE FORCEINLINE IntVector3& operator-=(const IntVector3& Other);
		HOST_DEVICE FORCEINLINE IntVector3& operator*=(const IntVector3& Other);
		HOST_DEVICE FORCEINLINE IntVector3& operator/=(const IntVector3& Other);
		HOST_DEVICE FORCEINLINE IntVector3& operator*=(const int& Value);
		HOST_DEVICE FORCEINLINE IntVector3& operator/=(const int& Value);

		HOST_DEVICE inline friend IntVector3 operator*(int Value, const IntVector3 &Vector);
		HOST_DEVICE inline friend IntVector3 operator/(int Value, const IntVector3 &Vector);
	};

	typedef Vector3 Point3;

}

#include "Math/IntVector3.inl"

