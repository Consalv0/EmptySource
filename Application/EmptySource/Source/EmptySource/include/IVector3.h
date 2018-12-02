#pragma once

struct FVector2;
struct FVector3;
struct FVector4;

struct IVector3 {
public:
	union {
		struct { int x, y, z; };
		struct { int r, g, b; };
	};

	FORCEINLINE IVector3();
	FORCEINLINE IVector3(const IVector3& Vector);
	FORCEINLINE IVector3(const FVector2& Vector);
	FORCEINLINE IVector3(const FVector3& Vector);
	FORCEINLINE IVector3(const FVector4& Vector);
	FORCEINLINE IVector3(const int& Value);
	FORCEINLINE IVector3(const int& x, const int& y, const int& z);
	FORCEINLINE IVector3(const int& x, const int& y);

	inline float Magnitude() const;
	inline int MagnitudeSquared() const;

	FORCEINLINE IVector3 Cross(const IVector3& Other) const;
	FORCEINLINE int Dot(const IVector3& Other) const;

	inline FVector3 FloatVector3() const;
	inline int & operator[](unsigned int i);
	inline int const& operator[](unsigned int i) const;
	inline const int* PointerToValue() const;

	FORCEINLINE bool operator==(const IVector3& Other);
	FORCEINLINE bool operator!=(const IVector3& Other);

	FORCEINLINE IVector3 operator+(const IVector3& Other) const;
	FORCEINLINE IVector3 operator-(const IVector3& Other) const;
	FORCEINLINE IVector3 operator-(void) const;
	FORCEINLINE IVector3 operator*(const IVector3& Other) const;
	FORCEINLINE IVector3 operator/(const IVector3& Other) const;
	FORCEINLINE IVector3 operator*(const int& Value) const;
	FORCEINLINE IVector3 operator/(const int& Value) const;

	FORCEINLINE IVector3& operator+=(const IVector3& Other);
	FORCEINLINE IVector3& operator-=(const IVector3& Other);
	FORCEINLINE IVector3& operator*=(const IVector3& Other);
	FORCEINLINE IVector3& operator/=(const IVector3& Other);
	FORCEINLINE IVector3& operator*=(const int& Value);
	FORCEINLINE IVector3& operator/=(const int& Value);
};

#include "..\include\IVector3.inl"