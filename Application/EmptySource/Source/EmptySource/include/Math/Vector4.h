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

	HOST_DEVICE FORCEINLINE Vector4();
	HOST_DEVICE FORCEINLINE Vector4(const Vector2& Vector);
	HOST_DEVICE FORCEINLINE Vector4(const Vector3& Vector);
	HOST_DEVICE FORCEINLINE Vector4(const Vector3& Vector, const float& w);
	HOST_DEVICE FORCEINLINE Vector4(const Vector4& Vector);
	HOST_DEVICE FORCEINLINE Vector4(const float& Value);
	HOST_DEVICE FORCEINLINE Vector4(const float& x, const float& y, const float& z);
	HOST_DEVICE FORCEINLINE Vector4(const float& x, const float& y, const float& z, const float& w);
	
	HOST_DEVICE inline float Magnitude() const;
	HOST_DEVICE inline float MagnitudeSquared() const;
	HOST_DEVICE inline void Normalize();
	HOST_DEVICE inline Vector4 Normalized() const;
	
	HOST_DEVICE FORCEINLINE float Dot(const Vector4& Other) const;
	HOST_DEVICE FORCEINLINE static Vector4 Lerp(const Vector4& Start, const Vector4& End, float t); 
	
	HOST_DEVICE inline float & operator[](unsigned int i);
	HOST_DEVICE inline float const& operator[](unsigned int i) const;
	HOST_DEVICE inline const float* PointerToValue() const;
	
	HOST_DEVICE FORCEINLINE bool operator==(const Vector4& Other) const;
	HOST_DEVICE FORCEINLINE bool operator!=(const Vector4& Other) const;
	
	HOST_DEVICE FORCEINLINE Vector4 operator+(const Vector4& Other) const;
	HOST_DEVICE FORCEINLINE Vector4 operator-(const Vector4& Other) const;
	HOST_DEVICE FORCEINLINE Vector4 operator-(void) const;
	HOST_DEVICE FORCEINLINE Vector4 operator*(const float& Value) const;
	HOST_DEVICE FORCEINLINE Vector4 operator/(const float& Value) const;
	HOST_DEVICE FORCEINLINE Vector4 operator*(const Vector4& Other) const;
	HOST_DEVICE FORCEINLINE Vector4 operator/(const Vector4& Other) const;
	
	HOST_DEVICE FORCEINLINE Vector4& operator+=(const Vector4& Other);
	HOST_DEVICE FORCEINLINE Vector4& operator-=(const Vector4& Other);
	HOST_DEVICE FORCEINLINE Vector4& operator*=(const Vector4& Other);
	HOST_DEVICE FORCEINLINE Vector4& operator/=(const Vector4& Other);
	HOST_DEVICE FORCEINLINE Vector4& operator*=(const float& Value);
	HOST_DEVICE FORCEINLINE Vector4& operator/=(const float& Value);
};

#include "..\Math\Vector4.inl"