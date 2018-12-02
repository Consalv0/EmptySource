#pragma once

#include "..\include\SCoreTypes.h"

struct FVector3;
struct FVector4;

struct FVector2 {
public:
	union {
		struct { float x, y; };
		struct { float u, v; };
	};

	FORCEINLINE FVector2();
	FORCEINLINE FVector2(const FVector2& Vector);
	FORCEINLINE FVector2(const FVector3& Vector);
	FORCEINLINE FVector2(const FVector4& Vector);
	FORCEINLINE FVector2(const float& Value);
	FORCEINLINE FVector2(const float& x, const float& y);

	inline float Magnitude() const;
	inline float MagnitudeSquared() const;
	inline void Normalize();
	inline FVector2 Normalized() const;

	FORCEINLINE float Cross(const FVector2& Other) const;
	FORCEINLINE float Dot(const FVector2& Other) const;
	FORCEINLINE static FVector2 Lerp(const FVector2& Start, const FVector2& End, float t);

	inline float & operator[](unsigned int i);
	inline float const& operator[](unsigned int i) const;
	inline const float* PointerToValue() const;

	FORCEINLINE bool operator==(const FVector2& Other);
	FORCEINLINE bool operator!=(const FVector2& Other);

	FORCEINLINE FVector2 operator+(const FVector2& Other) const;
	FORCEINLINE FVector2 operator-(const FVector2& Other) const;
	FORCEINLINE FVector2 operator-(void) const;
	FORCEINLINE FVector2 operator*(const FVector2& Other) const;
	FORCEINLINE FVector2 operator/(const FVector2& Other) const;
	FORCEINLINE FVector2 operator*(const float& Value) const;
	FORCEINLINE FVector2 operator/(const float& Value) const;

	FORCEINLINE FVector2& operator+=(const FVector2& Other);
	FORCEINLINE FVector2& operator-=(const FVector2& Other);
	FORCEINLINE FVector2& operator*=(const FVector2& Other);
	FORCEINLINE FVector2& operator/=(const FVector2& Other);
	FORCEINLINE FVector2& operator*=(const float& Value);
	FORCEINLINE FVector2& operator/=(const float& Value);
};

#include "..\include\FVector2.inl"