#pragma once

#include "..\CoreTypes.h"

struct Vector2;
struct Vector3;

struct Vector4 {
public:
	union {
		struct { float x, y, z, w; };
		struct { float r, g, b, a; };
	};

	FORCEINLINE Vector4();
	FORCEINLINE Vector4(const Vector2& Vector);
	FORCEINLINE Vector4(const Vector3& Vector);
	FORCEINLINE Vector4(const Vector4& Vector);
	FORCEINLINE Vector4(const float& Value);
	FORCEINLINE Vector4(const float& x, const float& y, const float& z);
	FORCEINLINE Vector4(const float& x, const float& y, const float& z, const float& w);
	
	inline float Magnitude() const;
	inline float MagnitudeSquared() const;
	inline void Normalize();
	inline Vector4 Normalized() const;

	FORCEINLINE float Dot(const Vector4& Other) const;
	FORCEINLINE static Vector4 Lerp(const Vector4& Start, const Vector4& End, float t); 

	inline float & operator[](unsigned int i);
	inline float const& operator[](unsigned int i) const;
	inline const float* PointerToValue() const;

	FORCEINLINE bool operator==(const Vector4& Other);
	FORCEINLINE bool operator!=(const Vector4& Other);

	FORCEINLINE Vector4 operator+(const Vector4& Other) const;
	FORCEINLINE Vector4 operator-(const Vector4& Other) const;
	FORCEINLINE Vector4 operator-(void) const;
	FORCEINLINE Vector4 operator*(const float& Value) const;
	FORCEINLINE Vector4 operator/(const float& Value) const;
	FORCEINLINE Vector4 operator*(const Vector4& Other) const;
	FORCEINLINE Vector4 operator/(const Vector4& Other) const;

	FORCEINLINE Vector4& operator+=(const Vector4& Other);
	FORCEINLINE Vector4& operator-=(const Vector4& Other);
	FORCEINLINE Vector4& operator*=(const Vector4& Other);
	FORCEINLINE Vector4& operator/=(const Vector4& Other);
	FORCEINLINE Vector4& operator*=(const float& Value);
	FORCEINLINE Vector4& operator/=(const float& Value);
};

#include "..\Math\Vector4.inl"