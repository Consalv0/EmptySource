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

	HOST_DEVICE FORCEINLINE Vector2();
	HOST_DEVICE FORCEINLINE Vector2(const Vector2& Vector);
	HOST_DEVICE FORCEINLINE Vector2(const Vector3& Vector);
	HOST_DEVICE FORCEINLINE Vector2(const Vector4& Vector);
	HOST_DEVICE FORCEINLINE Vector2(const float& Value);
	HOST_DEVICE FORCEINLINE Vector2(const float& x, const float& y);
	
	HOST_DEVICE inline float Magnitude() const;
	HOST_DEVICE inline float MagnitudeSquared() const;
	HOST_DEVICE inline void Normalize();
	HOST_DEVICE inline Vector2 Normalized() const;
	
	HOST_DEVICE FORCEINLINE float Cross(const Vector2& Other) const;
	HOST_DEVICE FORCEINLINE float Dot(const Vector2& Other) const;
	HOST_DEVICE FORCEINLINE static Vector2 Lerp(const Vector2& Start, const Vector2& End, float t);
	
	HOST_DEVICE inline float & operator[](unsigned int i);
	HOST_DEVICE inline float const& operator[](unsigned int i) const;
	HOST_DEVICE inline const float* PointerToValue() const;
	
	HOST_DEVICE FORCEINLINE bool operator==(const Vector2& Other) const;
	HOST_DEVICE FORCEINLINE bool operator!=(const Vector2& Other) const;
	
	HOST_DEVICE FORCEINLINE Vector2 operator+(const Vector2& Other) const;
	HOST_DEVICE FORCEINLINE Vector2 operator-(const Vector2& Other) const;
	HOST_DEVICE FORCEINLINE Vector2 operator-(void) const;
	HOST_DEVICE FORCEINLINE Vector2 operator*(const Vector2& Other) const;
	HOST_DEVICE FORCEINLINE Vector2 operator/(const Vector2& Other) const;
	HOST_DEVICE FORCEINLINE Vector2 operator*(const float& Value) const;
	HOST_DEVICE FORCEINLINE Vector2 operator/(const float& Value) const;
	
	HOST_DEVICE FORCEINLINE Vector2& operator+=(const Vector2& Other);
	HOST_DEVICE FORCEINLINE Vector2& operator-=(const Vector2& Other);
	HOST_DEVICE FORCEINLINE Vector2& operator*=(const Vector2& Other);
	HOST_DEVICE FORCEINLINE Vector2& operator/=(const Vector2& Other);
	HOST_DEVICE FORCEINLINE Vector2& operator*=(const float& Value);
	HOST_DEVICE FORCEINLINE Vector2& operator/=(const float& Value);
};

#include "..\Math\Vector2.inl"