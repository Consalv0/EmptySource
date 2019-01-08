#pragma once

#include "..\CoreTypes.h"

struct Vector3;
struct Vector4;

struct Vector2 {
public:
	union {
		struct { float x, y; };
		struct { float u, v; };
	};

	FORCEINLINE Vector2();
	FORCEINLINE Vector2(const Vector2& Vector);
	FORCEINLINE Vector2(const Vector3& Vector);
	FORCEINLINE Vector2(const Vector4& Vector);
	FORCEINLINE Vector2(const float& Value);
	FORCEINLINE Vector2(const float& x, const float& y);

	inline float Magnitude() const;
	inline float MagnitudeSquared() const;
	inline void Normalize();
	inline Vector2 Normalized() const;

	FORCEINLINE float Cross(const Vector2& Other) const;
	FORCEINLINE float Dot(const Vector2& Other) const;
	FORCEINLINE static Vector2 Lerp(const Vector2& Start, const Vector2& End, float t);

	inline float & operator[](unsigned int i);
	inline float const& operator[](unsigned int i) const;
	inline const float* PointerToValue() const;

	FORCEINLINE bool operator==(const Vector2& Other) const;
	FORCEINLINE bool operator!=(const Vector2& Other) const;

	FORCEINLINE Vector2 operator+(const Vector2& Other) const;
	FORCEINLINE Vector2 operator-(const Vector2& Other) const;
	FORCEINLINE Vector2 operator-(void) const;
	FORCEINLINE Vector2 operator*(const Vector2& Other) const;
	FORCEINLINE Vector2 operator/(const Vector2& Other) const;
	FORCEINLINE Vector2 operator*(const float& Value) const;
	FORCEINLINE Vector2 operator/(const float& Value) const;

	FORCEINLINE Vector2& operator+=(const Vector2& Other);
	FORCEINLINE Vector2& operator-=(const Vector2& Other);
	FORCEINLINE Vector2& operator*=(const Vector2& Other);
	FORCEINLINE Vector2& operator/=(const Vector2& Other);
	FORCEINLINE Vector2& operator*=(const float& Value);
	FORCEINLINE Vector2& operator/=(const float& Value);

	inline WString ToString();
};

#include "..\Math\Vector2.inl"