#pragma once

struct FVector4;
struct FVector3;

struct FMatrix4x4 {

public:
	struct {
		FVector4 m0, m1, m2, m3;
	};

	FORCEINLINE FMatrix4x4();
	FORCEINLINE FMatrix4x4(const FMatrix4x4& Other);
	FORCEINLINE FMatrix4x4(const FVector4& Row0, const FVector4& Row1, const FVector4& Row2, const FVector4 Row3);

	inline static FMatrix4x4 Identity();
	inline static FMatrix4x4 Perspective(const float& Aperture, const float& Aspect, const float& Near, const float& Far);
	inline static FMatrix4x4 LookAt(const FVector3& Position, const FVector3& Direction, const FVector3& Up);
	inline static FMatrix4x4 Translate(const FVector3& Vector);

	inline void Transpose();
	inline FMatrix4x4 Transposed() const;
	inline FMatrix4x4 Inversed() const;

	inline FVector4 Row(const int& i) const;
	inline FVector4 Column(const int& i) const;

	inline FVector4 & operator[](unsigned int i);
	inline FVector4 const& operator[](unsigned int i) const;
	inline const float* PointerToValue(void) const;
	
	FORCEINLINE FMatrix4x4 operator*(const FMatrix4x4& Other) const;
	FORCEINLINE FVector4 operator*(const FVector4& Vector) const;
	FORCEINLINE FVector3 operator*(const FVector3& Vector) const;
	FORCEINLINE FMatrix4x4 operator*(const float& Value) const;
	FORCEINLINE FMatrix4x4 operator/(const float& Value) const;
};

#include "FMatrix4x4.inl"