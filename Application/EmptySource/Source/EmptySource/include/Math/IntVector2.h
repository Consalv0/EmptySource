#pragma once

#include "..\CoreTypes.h"

struct Vector2;
struct Vector3;
struct Vector4;
struct IntVector3;

struct IntVector2 {
public:
	union {
		struct { int x, y; };
		struct { int u, v; };
	};

	FORCEINLINE IntVector2();
	FORCEINLINE IntVector2(const IntVector2& Vector);
	FORCEINLINE IntVector2(const IntVector3& Vector);
	FORCEINLINE IntVector2(const Vector2& Vector);
	FORCEINLINE IntVector2(const Vector3& Vector);
	FORCEINLINE IntVector2(const Vector4& Vector);
	FORCEINLINE IntVector2(const int& Value);
	FORCEINLINE IntVector2(const int& x, const int& y);

	inline float Magnitude() const;
	inline int MagnitudeSquared() const;

	FORCEINLINE int Dot(const IntVector2& Other) const;

	inline Vector2 FloatVector2() const;
	inline int & operator[](unsigned int i);
	inline int const& operator[](unsigned int i) const;
	inline const int* PointerToValue() const;

	FORCEINLINE bool operator==(const IntVector2& Other);
	FORCEINLINE bool operator!=(const IntVector2& Other);

	FORCEINLINE IntVector2 operator+(const IntVector2& Other) const;
	FORCEINLINE IntVector2 operator-(const IntVector2& Other) const;
	FORCEINLINE IntVector2 operator-(void) const;
	FORCEINLINE IntVector2 operator*(const IntVector2& Other) const;
	FORCEINLINE IntVector2 operator/(const IntVector2& Other) const;
	FORCEINLINE IntVector2 operator*(const int& Value) const;
	FORCEINLINE IntVector2 operator/(const int& Value) const;

	FORCEINLINE IntVector2& operator+=(const IntVector2& Other);
	FORCEINLINE IntVector2& operator-=(const IntVector2& Other);
	FORCEINLINE IntVector2& operator*=(const IntVector2& Other);
	FORCEINLINE IntVector2& operator/=(const IntVector2& Other);
	FORCEINLINE IntVector2& operator*=(const int& Value);
	FORCEINLINE IntVector2& operator/=(const int& Value);
};

#include "..\Math\IntVector2.inl"