#pragma once

#include "..\include\SCoreTypes.h"

struct FVector2;
struct FVector4;

struct FVector3 {
public:
	union {
		struct { float x, y, z; };
		struct { float r, g, b; };
	};

	FORCEINLINE FVector3();
	FORCEINLINE FVector3(const FVector2& Vector);
	FORCEINLINE FVector3(const FVector3& Vector);
	FORCEINLINE FVector3(const FVector4& Vector);
	FORCEINLINE FVector3(const float& Value);
	FORCEINLINE FVector3(const float& x, const float& y, const float& z);
	FORCEINLINE FVector3(const float& x, const float& y);

	inline float Magnitude() const;
	inline float MagnitudeSquared() const;
	inline void Normalize();
	inline FVector3 Normalized() const;

	FORCEINLINE FVector3 Cross(const FVector3& Other) const;
	FORCEINLINE float Dot(const FVector3& Other) const;
	FORCEINLINE static FVector3 Lerp(const FVector3& start, const FVector3& end, float t);

	inline FVector2 Vector2() const;
	inline float & operator[](unsigned int i);
	inline float const& operator[](unsigned int i) const;
	inline const float* PointerToValue() const;

	FORCEINLINE bool operator==(const FVector3& Other);
	FORCEINLINE bool operator!=(const FVector3& Other);

	FORCEINLINE FVector3 operator+(const FVector3& Other) const;
	FORCEINLINE FVector3 operator-(const FVector3& Other) const;
	FORCEINLINE FVector3 operator-(void) const;
	FORCEINLINE FVector3 operator*(const FVector3& Other) const;
	FORCEINLINE FVector3 operator/(const FVector3& Other) const;
	FORCEINLINE FVector3 operator*(const float& Value) const;
	FORCEINLINE FVector3 operator/(const float& Value) const;

	FORCEINLINE FVector3& operator+=(const FVector3& Other);
	FORCEINLINE FVector3& operator-=(const FVector3& Other);
	FORCEINLINE FVector3& operator*=(const FVector3& Other);
	FORCEINLINE FVector3& operator/=(const FVector3& Other);
	FORCEINLINE FVector3& operator*=(const float& Value);
	FORCEINLINE FVector3& operator/=(const float& Value);
};

#include "..\include\FVector3.inl"
