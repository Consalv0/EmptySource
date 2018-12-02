#pragma once

#include "..\include\SCoreTypes.h"
#include "..\include\FVector2.h"
#include "..\include\FVector3.h"

struct FVector4 {
public:
	union {
		struct { float x, y, z, w; };
		struct { float r, g, b, a; };
	};

	FORCEINLINE FVector4();
	FORCEINLINE FVector4(const FVector2& Vector);
	FORCEINLINE FVector4(const FVector3& Vector);
	FORCEINLINE FVector4(const FVector4& Vector);
	FORCEINLINE FVector4(const float& Value);
	FORCEINLINE FVector4(const float& x, const float& y, const float& z);
	FORCEINLINE FVector4(const float& x, const float& y, const float& z, const float& w);
	
	inline float Magnitude() const;
	inline float MagnitudeSquared() const;
	inline void Normalize();
	inline FVector4 Normalized() const;

	FORCEINLINE float Dot(const FVector4& Other) const;
	FORCEINLINE static FVector4 Lerp(const FVector4& Start, const FVector4& End, float t); 

	inline FVector3 Vector3() const;
	inline FVector2 Vector2() const;
	inline float & operator[](unsigned int i);
	inline float const& operator[](unsigned int i) const;
	inline const float* PointerToValue() const;

	FORCEINLINE bool operator==(const FVector4& Other);
	FORCEINLINE bool operator!=(const FVector4& Other);

	FORCEINLINE FVector4 operator+(const FVector4& Other) const;
	FORCEINLINE FVector4 operator-(const FVector4& Other) const;
	FORCEINLINE FVector4 operator-(void) const;
	FORCEINLINE FVector4 operator*(const float& Value) const;
	FORCEINLINE FVector4 operator/(const float& Value) const;
	FORCEINLINE FVector4 operator*(const FVector4& Other) const;
	FORCEINLINE FVector4 operator/(const FVector4& Other) const;

	FORCEINLINE FVector4& operator+=(const FVector4& Other);
	FORCEINLINE FVector4& operator-=(const FVector4& Other);
	FORCEINLINE FVector4& operator*=(const FVector4& Other);
	FORCEINLINE FVector4& operator/=(const FVector4& Other);
	FORCEINLINE FVector4& operator*=(const float& Value);
	FORCEINLINE FVector4& operator/=(const float& Value);
};

#include "FVector4.inl"