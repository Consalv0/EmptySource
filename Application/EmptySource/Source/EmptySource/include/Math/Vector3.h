#pragma once

#include "..\CoreTypes.h"

struct IntVector3;
struct Vector2;
struct Vector4;

struct Vector3 {
public:
	union {
		struct { float x, y, z; };
		struct { float r, g, b; };
	};

	FORCEINLINE Vector3();
	FORCEINLINE Vector3(const Vector2& Vector);
	FORCEINLINE Vector3(const IntVector3& Vector);
	FORCEINLINE Vector3(const Vector3& Vector);
	FORCEINLINE Vector3(const Vector4& Vector);
	FORCEINLINE Vector3(const float& Value);
	FORCEINLINE Vector3(const float& x, const float& y, const float& z);
	FORCEINLINE Vector3(const float& x, const float& y);

	inline float Magnitude() const;
	inline float MagnitudeSquared() const;
	inline void Normalize();
	inline Vector3 Normalized() const;

	FORCEINLINE Vector3 Cross(const Vector3& Other) const;
	FORCEINLINE float Dot(const Vector3& Other) const;
	FORCEINLINE static Vector3 Lerp(const Vector3& start, const Vector3& end, float t);

	inline float & operator[](unsigned int i);
	inline float const& operator[](unsigned int i) const;
	inline const float* PointerToValue() const;

	FORCEINLINE bool operator==(const Vector3& Other);
	FORCEINLINE bool operator!=(const Vector3& Other);

	FORCEINLINE Vector3 operator+(const Vector3& Other) const;
	FORCEINLINE Vector3 operator-(const Vector3& Other) const;
	FORCEINLINE Vector3 operator-(void) const;
	FORCEINLINE Vector3 operator*(const Vector3& Other) const;
	FORCEINLINE Vector3 operator/(const Vector3& Other) const;
	FORCEINLINE Vector3 operator*(const float& Value) const;
	FORCEINLINE Vector3 operator/(const float& Value) const;

	FORCEINLINE Vector3& operator+=(const Vector3& Other);
	FORCEINLINE Vector3& operator-=(const Vector3& Other);
	FORCEINLINE Vector3& operator*=(const Vector3& Other);
	FORCEINLINE Vector3& operator/=(const Vector3& Other);
	FORCEINLINE Vector3& operator*=(const float& Value);
	FORCEINLINE Vector3& operator/=(const float& Value);

	inline WString ToString();
};

#include "..\Math\Vector3.inl"
