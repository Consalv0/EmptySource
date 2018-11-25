
#include <math.h>
#include <stdexcept>

#include "..\include\FVector3.h"
#include "..\include\FVector4.h"
#include "..\include\FMatrix4x4.h"

FMatrix4x4::FMatrix4x4() {
	m0[0] = 1; m0[1] = 0; m0[2] = 0; m0[3] = 0;
	m1[0] = 0; m1[1] = 1; m1[2] = 0; m1[3] = 0;
	m2[0] = 0; m2[1] = 0; m2[2] = 1; m2[3] = 0;
	m3[0] = 0; m3[1] = 0; m3[2] = 0; m3[3] = 1;
}

FMatrix4x4::FMatrix4x4(const FMatrix4x4 & Other) 
	: m0(Other.m0), m1(Other.m1), m2(Other.m2), m3(Other.m3) {
	
}

FMatrix4x4::FMatrix4x4(const FVector4 & Row0, const FVector4 & Row1, const FVector4 & Row2, const FVector4 Row3) {
	m0[0] = Row0.x; m0[1] = Row0.y; m0[2] = Row0.z; m0[3] = Row0.w;
	m1[0] = Row1.x; m1[1] = Row1.y; m1[2] = Row1.z; m1[3] = Row1.w;
	m2[0] = Row2.x; m2[1] = Row2.y; m2[2] = Row2.z; m2[3] = Row2.w;
	m3[0] = Row3.x; m3[1] = Row3.y; m3[2] = Row3.z; m3[3] = Row3.w;
}

FMatrix4x4 FMatrix4x4::Identity() {
	return FMatrix4x4();
}

FMatrix4x4 FMatrix4x4::Perspective(const float & FOV, const float & Aspect, const float & Near, const float & Far) {
	FMatrix4x4 Result = FMatrix4x4();
	
	float const TangentHalfFOV = tan(FOV / 2.F);

	Result.m0[0] = 1.F / (Aspect * TangentHalfFOV);
	Result.m1[1] = 1.F / (TangentHalfFOV);
	Result.m2[2] = -(Far + Near) / (Far - Near);
	Result.m2[3] = -1.F;
	Result.m3[2] = -(2.F * Far * Near) / (Far - Near);
	
	return Result;
}

FMatrix4x4 FMatrix4x4::LookAt(const FVector3 & Eye, const FVector3 & Target, const FVector3 & Up) {
	FMatrix4x4 Result = FMatrix4x4();

	FVector3 const f((Target - Eye).Normalized());
	FVector3 const s(f.Cross(Up).Normalized());
	FVector3 const u(s.Cross(f));

	Result.m0[0] = s.x;
	Result.m1[0] = s.y;
	Result.m2[0] = s.z;
	Result.m0[1] = u.x;
	Result.m1[1] = u.y;
	Result.m2[1] = u.z;
	Result.m0[2] = -f.x;
	Result.m1[2] = -f.y;
	Result.m2[2] = -f.z;
	Result.m3[0] = -s.Dot(Eye);
	Result.m3[1] = -u.Dot(Eye);
	Result.m3[2] = f.Dot(Eye);
	return Result;
}

void FMatrix4x4::Transpose() {
	FMatrix4x4 Result = FMatrix4x4(Column(0), Column(1), Column(2), Column(3));
	*this = Result;
}

FMatrix4x4 FMatrix4x4::Transposed() const {
	return FMatrix4x4(Column(0), Column(1), Column(2), Column(3));
}

FVector4 FMatrix4x4::Row(const int & i) const {
	switch (i) {
		case 0: return m0;
		case 1: return m1;
		case 2: return m2;
		case 3: return m3;
	}

	return FVector4();
}

FVector4 FMatrix4x4::Column(const int & i) const {
	switch (i) {
		case 0: return FVector4(m0[0], m1[0], m2[0], m3[0]);
		case 1: return FVector4(m0[1], m1[1], m2[1], m3[1]);
		case 2: return FVector4(m0[2], m1[2], m2[2], m3[2]);
		case 3: return FVector4(m0[3], m1[3], m2[3], m3[3]);
	}

	return FVector4();
}

FVector4 & FMatrix4x4::operator[](unsigned int i) {
	switch (i) {
		case 0:  return m0;
		case 1:  return m1;
		case 2:  return m2;
		case 3:  return m3;
		default: return m3;
	}
}

FVector4 const & FMatrix4x4::operator[](unsigned int i) const {
	switch (i) {
		case 0:  return m0;
		case 1:  return m1;
		case 2:  return m2;
		case 3:  return m3;
		default: return m3;
	}
}

FMatrix4x4 FMatrix4x4::operator*(const FMatrix4x4 & Other) const {
	FMatrix4x4 Result = FMatrix4x4();

	FVector4 OtherCol0 = Other.Column(0), OtherCol1 = Other.Column(1), OtherCol2 = Other.Column(2), OtherCol3 = Other.Column(3);

	Result.m0[0] = m0.Dot(OtherCol0);
	Result.m1[0] = m1.Dot(OtherCol0);
	Result.m2[0] = m2.Dot(OtherCol0);
	Result.m3[0] = m3.Dot(OtherCol0);
	
	Result.m0[1] = m0.Dot(OtherCol1);
	Result.m1[1] = m1.Dot(OtherCol1);
	Result.m2[1] = m2.Dot(OtherCol1);
	Result.m3[1] = m3.Dot(OtherCol1);

	Result.m0[2] = m0.Dot(OtherCol2);
	Result.m1[2] = m1.Dot(OtherCol2);
	Result.m2[2] = m2.Dot(OtherCol2);
	Result.m3[2] = m3.Dot(OtherCol2);

	Result.m0[3] = m0.Dot(OtherCol3);
	Result.m1[3] = m1.Dot(OtherCol3);
	Result.m2[3] = m2.Dot(OtherCol3);
	Result.m3[3] = m3.Dot(OtherCol3);
	
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
	return &m0[0];
}
