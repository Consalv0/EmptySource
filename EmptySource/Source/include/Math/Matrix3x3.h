#pragma once

struct Vector3;
struct Matrix4x4;

struct Matrix3x3 {
public:
	struct {
		Vector3 m0, m1, m2;
	};

	HOST_DEVICE FORCEINLINE Matrix3x3();
	HOST_DEVICE FORCEINLINE Matrix3x3(const Matrix3x3& Matrix);
	HOST_DEVICE FORCEINLINE Matrix3x3(const Matrix4x4& Matrix);
	HOST_DEVICE FORCEINLINE Matrix3x3(const Vector3& Row0, const Vector3& Row1, const Vector3& Row2);

	HOST_DEVICE inline static Matrix3x3 Identity();

	HOST_DEVICE inline void Transpose();
	HOST_DEVICE inline Matrix3x3 Transposed() const;

	HOST_DEVICE inline Vector3 Row(const int& i) const;
	HOST_DEVICE inline Vector3 Column(const int& i) const;

	HOST_DEVICE inline Vector3 & operator[](unsigned int i);
	HOST_DEVICE inline Vector3 const& operator[](unsigned int i) const;
	HOST_DEVICE inline const float* PointerToValue(void) const;

	HOST_DEVICE FORCEINLINE Matrix3x3 operator*(const Matrix3x3& Other) const;
	HOST_DEVICE FORCEINLINE Vector3 operator*(const Vector3& Vector) const;
};

#include "../Math/Matrix3x3.inl"
