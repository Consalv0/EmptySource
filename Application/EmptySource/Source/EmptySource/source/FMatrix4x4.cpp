
#include <math.h>
#include <stdexcept>

#include "..\include\FVector3.h"
#include "..\include\FVector4.h"
#include "..\include\FMatrix4x4.h"

FMatrix4x4::FMatrix4x4() {
	m00 = 1; m01 = 0; m02 = 0; m03 = 0;
	m10 = 0; m11 = 1; m12 = 0; m13 = 0;
	m20 = 0; m21 = 0; m22 = 1; m23 = 0;
	m30 = 0; m31 = 0; m32 = 0; m33 = 1;
}

FMatrix4x4::FMatrix4x4(const FVector4 & Row0, const FVector4 & Row1, const FVector4 & Row2, const FVector4 Row3) {
	m00 = Row0.x; m01 = Row0.y; m02 = Row0.z; m03 = Row0.w;
	m10 = Row1.x; m11 = Row1.y; m12 = Row1.z; m13 = Row1.w;
	m20 = Row2.x; m21 = Row2.y; m22 = Row2.z; m23 = Row2.w;
	m30 = Row3.x; m31 = Row3.y; m32 = Row3.z; m33 = Row3.w;
}

FMatrix4x4 FMatrix4x4::Identity() {
	return FMatrix4x4();
}

FMatrix4x4 FMatrix4x4::Perspective(const float & FOV, const float & Aspect, const float & Near, const float & Far) {
	FMatrix4x4 Result = FMatrix4x4();
	
	float TangentHalfFOV = tan(FOV / 2.F);

	Result.m00 = 1.F / (Aspect * TangentHalfFOV);
	Result.m11 = 1.F / (TangentHalfFOV);
	Result.m22 = -(Far + Near) / (Far - Near);
	Result.m23 = -1.F;
	Result.m32 = -(2.F * Far * Near) / (Far - Near);
	
	return Result;
}

FMatrix4x4 FMatrix4x4::LookAt(const FVector3 & Eye, const FVector3 & Target, const FVector3 & Up) {
	FMatrix4x4 Result = FMatrix4x4();

	FVector3 const f((Target - Eye).Normalized());
	FVector3 const s(f.Cross(Up).Normalized());
	FVector3 const u(s.Cross(f));

	Result.m00 = s.x;
	Result.m10 = s.y;
	Result.m20 = s.z;
	Result.m01 = u.x;
	Result.m11 = u.y;
	Result.m21 = u.z;
	Result.m02 = -f.x;
	Result.m12 = -f.y;
	Result.m22 = -f.z;
	Result.m30 = -s.Dot(Eye);
	Result.m31 = -u.Dot(Eye);
	Result.m32 = f.Dot(Eye);
	return Result;
}

void FMatrix4x4::Transpose() {
	FMatrix4x4 Result = FMatrix4x4(Result.Column(0), Result.Column(1), Result.Column(2), Result.Column(3));
	*this = Result;
}

FMatrix4x4 FMatrix4x4::Transposed() const {
	FMatrix4x4 Result = FMatrix4x4(Result.Column(0), Result.Column(1), Result.Column(2), Result.Column(3));
	return Result;
}

FVector4 FMatrix4x4::Row(const int & i) const {
	switch (i)
	{
	case 0:
		return FVector4(m00, m01, m02, m03);
	case 1:
		return FVector4(m10, m11, m12, m13);
	case 2:
		return FVector4(m20, m21, m22, m23);
	case 3:
		return FVector4(m30, m31, m32, m33);
	}

	return FVector4();
}

FVector4 FMatrix4x4::Column(const int & i) const {
	switch (i)
	{
	case 0:
		return FVector4(m00, m10, m20, m30);
	case 1:
		return FVector4(m01, m11, m21, m31);
	case 2:
		return FVector4(m02, m12, m22, m32);
	case 3:
		return FVector4(m03, m13, m23, m33);
	}

	return FVector4();
}

FMatrix4x4 FMatrix4x4::operator*(const FMatrix4x4 & Other) const {
	FMatrix4x4 Result = FMatrix4x4();

	FVector4 OtherCol0 = Other.Column(0), OtherCol1 = Other.Column(1), OtherCol2 = Other.Column(2), OtherCol3 = Other.Column(3);
	FVector4 Row0 = Row(0), Row1 = Row(1), Row2 = Row(2), Row3 = Row(3);

	Result.m00 = Row0.Dot(OtherCol0);
	Result.m10 = Row1.Dot(OtherCol0);
	Result.m20 = Row2.Dot(OtherCol0);
	Result.m30 = Row3.Dot(OtherCol0);
	
	Result.m01 = Row0.Dot(OtherCol1);
	Result.m11 = Row1.Dot(OtherCol1);
	Result.m21 = Row2.Dot(OtherCol1);
	Result.m31 = Row3.Dot(OtherCol1);

	Result.m02 = Row0.Dot(OtherCol2);
	Result.m12 = Row1.Dot(OtherCol2);
	Result.m22 = Row2.Dot(OtherCol2);
	Result.m32 = Row3.Dot(OtherCol2);
	
	Result.m03 = Row0.Dot(OtherCol3);
	Result.m13 = Row1.Dot(OtherCol3);
	Result.m23 = Row2.Dot(OtherCol3);
	Result.m33 = Row3.Dot(OtherCol3);
	
	return Result;
}

FVector4 FMatrix4x4::operator*(const FVector4 & Vector) const {
	FVector4 Result(
		Row(0).Dot(Vector),
		Row(1).Dot(Vector),
		Row(2).Dot(Vector),
		Row(3).Dot(Vector)
	);

	return Result;
}

FVector3 FMatrix4x4::operator*(const FVector3 & Vector) const {
	FVector3 Result(
		Row(0).Dot(Vector),
		Row(1).Dot(Vector),
		Row(2).Dot(Vector)
	);

	return Result;
}

const float * FMatrix4x4::PoiterToValue(void) const {
	return &m00;
}
