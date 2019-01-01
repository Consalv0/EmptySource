#pragma once

#include "..\CoreTypes.h"

struct Vector2;
struct Vector3;
struct Vector4;

struct IntVector3 {
public:
	union {
		struct { int x, y, z; };
		struct { int r, g, b; };
	};

	FORCEINLINE IntVector3();
	FORCEINLINE IntVector3(const IntVector3& Vector);
	FORCEINLINE IntVector3(const Vector2& Vector);
	FORCEINLINE IntVector3(const Vector3& Vector);
	FORCEINLINE IntVector3(const Vector4& Vector);
	FORCEINLINE IntVector3(const int& Value);
	FORCEINLINE IntVector3(const int& x, const int& y, const int& z);
	FORCEINLINE IntVector3(const int& x, const int& y);

	inline float Magnitude() const;
	inline int MagnitudeSquared() const;

	FORCEINLINE IntVector3 Cross(const IntVector3& Other) const;
	FORCEINLINE int Dot(const IntVector3& Other) const;

	inline Vector3 FloatVector3() const;
	inline int & operator[](unsigned int i);
	inline int const& operator[](unsigned int i) const;
	inline const int* PointerToValue() const;

	FORCEINLINE bool operator==(const IntVector3& Other);
	FORCEINLINE bool operator!=(const IntVector3& Other);

	FORCEINLINE IntVector3 operator+(const IntVector3& Other) const;
	FORCEINLINE IntVector3 operator-(const IntVector3& Other) const;
	FORCEINLINE IntVector3 operator-(void) const;
	FORCEINLINE IntVector3 operator*(const IntVector3& Other) const;
	FORCEINLINE IntVector3 operator/(const IntVector3& Other) const;
	FORCEINLINE IntVector3 operator*(const int& Value) const;
	FORCEINLINE IntVector3 operator/(const int& Value) const;

	FORCEINLINE IntVector3& operator+=(const IntVector3& Other);
	FORCEINLINE IntVector3& operator-=(const IntVector3& Other);
	FORCEINLINE IntVector3& operator*=(const IntVector3& Other);
	FORCEINLINE IntVector3& operator/=(const IntVector3& Other);
	FORCEINLINE IntVector3& operator*=(const int& Value);
	FORCEINLINE IntVector3& operator/=(const int& Value);

	inline WString ToString();
};

#include "..\Math\IntVector3.inl"