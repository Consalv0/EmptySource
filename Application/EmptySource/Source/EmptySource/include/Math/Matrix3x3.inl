
#include <math.h>
#include <stdexcept>
#include "..\Math\Vector3.h"
#include "..\Math\Matrix4x4.h"
#include "..\Math\Matrix3x3.h"

FORCEINLINE FMatrix3x3::FMatrix3x3() {
	m0[0] = 1; m0[1] = 0; m0[2] = 0;
	m1[0] = 0; m1[1] = 1; m1[2] = 0;
	m2[0] = 0; m2[1] = 0; m2[2] = 1;
}

FORCEINLINE FMatrix3x3::FMatrix3x3(const FMatrix3x3 & Matrix)
	: m0(Matrix.m0), m1(Matrix.m1), m2(Matrix.m2) {
}

FORCEINLINE FMatrix3x3::FMatrix3x3(const Matrix4x4 & Matrix) {
	FMatrix3x3(Matrix.Row(0), Matrix.Row(1), Matrix.Row(2));
}

FORCEINLINE FMatrix3x3::FMatrix3x3(const Vector3 & Row0, const Vector3 & Row1, const Vector3 & Row2) {
	m0[0] = Row0.x, m0[1] = Row0.y, m0[2] = Row0.z;
	m1[0] = Row1.x, m1[1] = Row1.y, m1[2] = Row1.z;
	m2[0] = Row2.x, m2[1] = Row2.y, m2[2] = Row2.z;
}

inline FMatrix3x3 FMatrix3x3::Identity() {
	return FMatrix3x3();
}

inline void FMatrix3x3::Transpose() {
	*this = FMatrix3x3(Column(0), Column(1), Column(2));
}

inline FMatrix3x3 FMatrix3x3::Transposed() const {
	return FMatrix3x3(Column(0), Column(1), Column(2));
}

inline Vector3 FMatrix3x3::Row(const int & i) const {
	switch (i) {
		case 0: return Vector3(m0[0], m0[1], m0[2]);
		case 1: return Vector3(m1[0], m1[1], m1[2]);
		case 2: return Vector3(m2[0], m2[1], m2[2]);
	}

	return Vector3();
}

inline Vector3 FMatrix3x3::Column(const int & i) const {
	switch (i) {
		case 0: return Vector3(m0[0], m1[0], m2[0]);
		case 1: return Vector3(m0[1], m1[1], m2[1]);
		case 2: return Vector3(m0[2], m1[2], m2[2]);
	}

	return Vector3();
}

inline Vector3 & FMatrix3x3::operator[](unsigned int i) {
	switch (i) {
		case 0:  return m0;
		case 1:  return m1;
		case 2:  return m2;
		default: return m2;
	}
}

inline Vector3 const & FMatrix3x3::operator[](unsigned int i) const {
	switch (i) {
		case 0:  return m0;
		case 1:  return m1;
		case 2:  return m2;
		default: return m2;
	}
}

FORCEINLINE FMatrix3x3 FMatrix3x3::operator*(const FMatrix3x3& Other) const {
	FMatrix3x3 ReturnMatrix = FMatrix3x3();
	Vector3 OtherCol0 = Other.Column(0), OtherCol1 = Other.Column(1), OtherCol2 = Other.Column(2);
	Vector3 Row0 = Row(0), Row1 = Row(1), Row2 = Row(2);
	
	ReturnMatrix.m0[0] = Row0.Dot(OtherCol0);
	ReturnMatrix.m1[0] = Row1.Dot(OtherCol0);
	ReturnMatrix.m2[0] = Row2.Dot(OtherCol0);
	
	ReturnMatrix.m0[1] = Row0.Dot(OtherCol1);
	ReturnMatrix.m1[1] = Row1.Dot(OtherCol1);
	ReturnMatrix.m2[1] = Row2.Dot(OtherCol1);
	
	ReturnMatrix.m0[2] = Row0.Dot(OtherCol2);
	ReturnMatrix.m1[2] = Row1.Dot(OtherCol2);
	ReturnMatrix.m2[2] = Row2.Dot(OtherCol2);

	return ReturnMatrix;
}

FORCEINLINE Vector3 FMatrix3x3::operator*(const Vector3& Vector) const {
	Vector3 result(
		Row(0).Dot(Vector),
		Row(1).Dot(Vector),
		Row(2).Dot(Vector)
	);

	return result;
}

inline const float * FMatrix3x3::PointerToValue(void) const {
	return &m0[0];
}
