#pragma once

struct FVector3;
struct FMatrix4x4;

struct FMatrix3x3 {
public:
	struct {
		FVector3 m0, m1, m2;
	};

	FORCEINLINE FMatrix3x3();
	FORCEINLINE FMatrix3x3(const FMatrix3x3& Matrix);
	FORCEINLINE FMatrix3x3(const FMatrix4x4& Matrix);
	FORCEINLINE FMatrix3x3(const FVector3& Row0, const FVector3& Row1, const FVector3& Row2);

	inline static FMatrix3x3 Identity();

	inline void Transpose();
	inline FMatrix3x3 Transposed() const;

	inline FVector3 Row(const int& i) const;
	inline FVector3 Column(const int& i) const;

	inline FVector3 & operator[](unsigned int i);
	inline FVector3 const& operator[](unsigned int i) const;
	inline const float* PointerToValue(void) const;

	FORCEINLINE FMatrix3x3 operator*(const FMatrix3x3& Other) const;
	FORCEINLINE FVector3 operator*(const FVector3& Vector) const;
};

#include "FMatrix3x3.inl"