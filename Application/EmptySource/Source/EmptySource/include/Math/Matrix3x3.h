#pragma once

struct Vector3;
struct Matrix4x4;

struct FMatrix3x3 {
public:
	struct {
		Vector3 m0, m1, m2;
	};

	FORCEINLINE FMatrix3x3();
	FORCEINLINE FMatrix3x3(const FMatrix3x3& Matrix);
	FORCEINLINE FMatrix3x3(const Matrix4x4& Matrix);
	FORCEINLINE FMatrix3x3(const Vector3& Row0, const Vector3& Row1, const Vector3& Row2);

	inline static FMatrix3x3 Identity();

	inline void Transpose();
	inline FMatrix3x3 Transposed() const;

	inline Vector3 Row(const int& i) const;
	inline Vector3 Column(const int& i) const;

	inline Vector3 & operator[](unsigned int i);
	inline Vector3 const& operator[](unsigned int i) const;
	inline const float* PointerToValue(void) const;

	FORCEINLINE FMatrix3x3 operator*(const FMatrix3x3& Other) const;
	FORCEINLINE Vector3 operator*(const Vector3& Vector) const;
};

#include "..\Math\Matrix3x3.inl"