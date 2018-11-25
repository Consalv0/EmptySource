
#include <math.h>
#include <stdexcept>
#include "..\include\FVector3.h"
#include "..\include\FVector4.h"
#include "..\include\FMatrix3x3.h"
#include "..\include\FMatrix4x4.h"

FMatrix3x3::FMatrix3x3() {
	m00 = 1; m01 = 0; m02 = 0;
	m10 = 0; m11 = 1; m12 = 0;
	m20 = 0; m21 = 0; m22 = 1;
}

FMatrix3x3::FMatrix3x3(const FMatrix4x4 & Matrix) {
	FMatrix3x3(Matrix.Row(0), Matrix.Row(1), Matrix.Row(2));
}

FMatrix3x3::FMatrix3x3(const FVector3 & Row0, const FVector3 & Row1, const FVector3 & Row2) {
	m00 = Row0.x, m01 = Row0.y, m02 = Row0.z;
	m10 = Row1.x, m11 = Row1.y, m12 = Row1.z;
	m20 = Row2.x, m21 = Row2.y, m22 = Row2.z;
}

FMatrix3x3 FMatrix3x3::Identity() {
	return FMatrix3x3();
}

void FMatrix3x3::Transpose() {
	FMatrix3x3 newMatrix = FMatrix3x3(Column(0), Column(1), Column(2));
	*this = newMatrix;
}

FVector3 FMatrix3x3::Row(const int & i) const {
	switch (i)
	{
	case 0:
		return FVector3(m00, m01, m02);
	case 1:
		return FVector3(m10, m11, m12);
	case 2:
		return FVector3(m20, m21, m22);
	}

	return FVector3();
}

FVector3 FMatrix3x3::Column(const int & i) const {
	switch (i)
	{
	case 0:
		return FVector3(m00, m10, m20);
	case 1:
		return FVector3(m01, m11, m21);
	case 2:
		return FVector3(m02, m12, m22);
	}

	return FVector3();
}

FMatrix3x3 FMatrix3x3::operator*(const FMatrix3x3& Other) const {
	FMatrix3x3 ReturnMatrix = FMatrix3x3();
	FVector3 OtherCol0 = Other.Column(0), OtherCol1 = Other.Column(1), OtherCol2 = Other.Column(2);
	FVector3 Row0 = Row(0), Row1 = Row(1), Row2 = Row(2);
	
	ReturnMatrix.m00 = Row0.Dot(OtherCol0);
	ReturnMatrix.m10 = Row1.Dot(OtherCol0);
	ReturnMatrix.m20 = Row2.Dot(OtherCol0);
	
	ReturnMatrix.m01 = Row0.Dot(OtherCol1);
	ReturnMatrix.m11 = Row1.Dot(OtherCol1);
	ReturnMatrix.m21 = Row2.Dot(OtherCol1);
	
	ReturnMatrix.m02 = Row0.Dot(OtherCol2);
	ReturnMatrix.m12 = Row1.Dot(OtherCol2);
	ReturnMatrix.m22 = Row2.Dot(OtherCol2);

	return ReturnMatrix;
}

FVector3 FMatrix3x3::operator*(const FVector3& Vector) const {
	FVector3 result(
		Row(0).Dot(Vector),
		Row(1).Dot(Vector),
		Row(2).Dot(Vector)
	);

	return result;
}

const float * FMatrix3x3::PoiterToValue(void) const {
	return &m00;
}
