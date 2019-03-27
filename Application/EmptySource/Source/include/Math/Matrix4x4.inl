
#include <math.h>
#include <stdexcept>

#include "../Math/Vector3.h"
#include "../Math/Vector4.h"
#include "../Math/Matrix4x4.h"

FORCEINLINE Matrix4x4::Matrix4x4() {
	m0[0] = 1; m0[1] = 0; m0[2] = 0; m0[3] = 0;
	m1[0] = 0; m1[1] = 1; m1[2] = 0; m1[3] = 0;
	m2[0] = 0; m2[1] = 0; m2[2] = 1; m2[3] = 0;
	m3[0] = 0; m3[1] = 0; m3[2] = 0; m3[3] = 1;
}

FORCEINLINE Matrix4x4::Matrix4x4(const Matrix4x4 & Other)
	: m0(Other.m0), m1(Other.m1), m2(Other.m2), m3(Other.m3) {
	
}

FORCEINLINE Matrix4x4::Matrix4x4(const Vector4 & Row0, const Vector4 & Row1, const Vector4 & Row2, const Vector4 Row3) {
	m0[0] = Row0.x; m0[1] = Row0.y; m0[2] = Row0.z; m0[3] = Row0.w;
	m1[0] = Row1.x; m1[1] = Row1.y; m1[2] = Row1.z; m1[3] = Row1.w;
	m2[0] = Row2.x; m2[1] = Row2.y; m2[2] = Row2.z; m2[3] = Row2.w;
	m3[0] = Row3.x; m3[1] = Row3.y; m3[2] = Row3.z; m3[3] = Row3.w;
}

inline Matrix4x4 Matrix4x4::Identity() {
	return Matrix4x4();
}

inline Matrix4x4 Matrix4x4::Perspective(const float & FOV, const float & Aspect, const float & Near, const float & Far) {
	Matrix4x4 Result = Matrix4x4();
	
	float const TangentHalfFOV = tan(FOV / 2.F);

	Result.m0[0] = 1.F / (Aspect * TangentHalfFOV);
	Result.m1[1] = 1.F / (TangentHalfFOV);
	Result.m2[2] = -(Far + Near) / (Far - Near);
	Result.m2[3] = -1.F;
	Result.m3[2] = -(2.F * Far * Near) / (Far - Near);
	Result.m3[3] = 0.F;
	
	return Result;
}

inline HOST_DEVICE Matrix4x4 Matrix4x4::Orthographic(const float & Left, const float & Right, const float & Bottom, const float & Top) {
	Matrix4x4 Result = Matrix4x4();
	Result.m0[0] = 2.F / (Right - Left);
	Result.m1[1] = 2.F / (Top - Bottom);
	Result.m3[0] = -(Right + Left) / (Right - Left);
	Result.m3[1] = -(Top + Bottom) / (Top - Bottom);
	return Result;
}

inline Matrix4x4 Matrix4x4::LookAt(const Vector3 & Eye, const Vector3 & Target, const Vector3 & Up) {
	Matrix4x4 Result = Matrix4x4();

	Vector3 const f((Target - Eye).Normalized());
	Vector3 const s(f.Cross(Up).Normalized());
	Vector3 const u(s.Cross(f));

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

inline Matrix4x4 Matrix4x4::Translation(const Vector3 & Vector) {
	Matrix4x4 Result = Matrix4x4();
	Result.w = Result.x * Vector.x + Result.y * Vector.y + Result.z * Vector.z + Result.w;
	return Result;
}

inline Matrix4x4 Matrix4x4::Scaling(const Vector3 & Vector) {
	Matrix4x4 Result = Matrix4x4();
	Result.m0 = Result.m0 * Vector[0];
	Result.m1 = Result.m1 * Vector[1];
	Result.m2 = Result.m2 * Vector[2];
	return Result;
}

inline Matrix4x4 Matrix4x4::Rotation(const Vector3 & Axis, const float & Angle) {
	float const cosA = cos(Angle);
	float const sinA = sin(Angle);

	Vector3 AxisN(Axis.Normalized());
	Vector3 Temp((1.F - cosA) * AxisN);

	Matrix4x4 Rotation;
	Rotation[0][0] = cosA + Temp[0] * AxisN[0];
	Rotation[0][1] = Temp[0] * AxisN[1] + sinA * AxisN[2];
	Rotation[0][2] = Temp[0] * AxisN[2] - sinA * AxisN[1];

	Rotation[1][0] = Temp[1] * AxisN[0] - sinA * AxisN[2];
	Rotation[1][1] = cosA + Temp[1] * AxisN[1];
	Rotation[1][2] = Temp[1] * AxisN[2] + sinA * AxisN[0];

	Rotation[2][0] = Temp[2] * AxisN[0] + sinA * AxisN[1];
	Rotation[2][1] = Temp[2] * AxisN[1] - sinA * AxisN[0];
	Rotation[2][2] = cosA + Temp[2] * AxisN[2];

	Vector4 m0 = { 1, 0, 0, 0 };
	Vector4 m1 = { 0, 1, 0, 0 };
	Vector4 m2 = { 0, 0, 1, 0 };

	Matrix4x4 Result;
	Result[0] = m0 * Rotation[0][0] + m1 * Rotation[0][1] + m2 * Rotation[0][2];
	Result[1] = m0 * Rotation[1][0] + m1 * Rotation[1][1] + m2 * Rotation[1][2];
	Result[2] = m0 * Rotation[2][0] + m1 * Rotation[2][1] + m2 * Rotation[2][2];
	Result[3] = {0, 0, 0, 1};
	return Result;
}

inline void Matrix4x4::Transpose() {
	Matrix4x4 Result = Matrix4x4(Column(0), Column(1), Column(2), Column(3));
	*this = Result;
}

inline Matrix4x4 Matrix4x4::Transposed() const {
	return Matrix4x4(Column(0), Column(1), Column(2), Column(3));
}

inline Matrix4x4 Matrix4x4::Inversed() const {
	float Coef00 = m2[2] * m3[3] - m3[2] * m2[3];
	float Coef02 = m1[2] * m3[3] - m3[2] * m1[3];
	float Coef03 = m1[2] * m2[3] - m2[2] * m1[3];

	float Coef04 = m2[1] * m3[3] - m3[1] * m2[3];
	float Coef06 = m1[1] * m3[3] - m3[1] * m1[3];
	float Coef07 = m1[1] * m2[3] - m2[1] * m1[3];

	float Coef08 = m2[1] * m3[2] - m3[1] * m2[2];
	float Coef10 = m1[1] * m3[2] - m3[1] * m1[2];
	float Coef11 = m1[1] * m2[2] - m2[1] * m1[2];

	float Coef12 = m2[0] * m3[3] - m3[0] * m2[3];
	float Coef14 = m1[0] * m3[3] - m3[0] * m1[3];
	float Coef15 = m1[0] * m2[3] - m2[0] * m1[3];

	float Coef16 = m2[0] * m3[2] - m3[0] * m2[2];
	float Coef18 = m1[0] * m3[2] - m3[0] * m1[2];
	float Coef19 = m1[0] * m2[2] - m2[0] * m1[2];

	float Coef20 = m2[0] * m3[1] - m3[0] * m2[1];
	float Coef22 = m1[0] * m3[1] - m3[0] * m1[1];
	float Coef23 = m1[0] * m2[1] - m2[0] * m1[1];

	Vector4 Fac0(Coef00, Coef00, Coef02, Coef03);
	Vector4 Fac1(Coef04, Coef04, Coef06, Coef07);
	Vector4 Fac2(Coef08, Coef08, Coef10, Coef11);
	Vector4 Fac3(Coef12, Coef12, Coef14, Coef15);
	Vector4 Fac4(Coef16, Coef16, Coef18, Coef19);
	Vector4 Fac5(Coef20, Coef20, Coef22, Coef23);

	Vector4 Vec0(m1[0], m0[0], m0[0], m0[0]);
	Vector4 Vec1(m1[1], m0[1], m0[1], m0[1]);
	Vector4 Vec2(m1[2], m0[2], m0[2], m0[2]);
	Vector4 Vec3(m1[3], m0[3], m0[3], m0[3]);

	Vector4 Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
	Vector4 Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
	Vector4 Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
	Vector4 Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

	Vector4 SignA(+1, -1, +1, -1);
	Vector4 SignB(-1, +1, -1, +1);

	Matrix4x4 Result(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);

	// Vector4 Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

	Vector4 Dot0(Row(0) * Result.Column(0));
	float Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

	float OneOverDeterminant = float(1) / Dot1;

	return Result * OneOverDeterminant;
}

inline Vector4 Matrix4x4::Row(const int & i) const {
	switch (i) {
		case 0: return m0;
		case 1: return m1;
		case 2: return m2;
		case 3: return m3;
	}

	return Vector4();
}

inline Vector4 Matrix4x4::Column(const int & i) const {
	switch (i) {
		case 0: return Vector4(m0[0], m1[0], m2[0], m3[0]);
		case 1: return Vector4(m0[1], m1[1], m2[1], m3[1]);
		case 2: return Vector4(m0[2], m1[2], m2[2], m3[2]);
		case 3: return Vector4(m0[3], m1[3], m2[3], m3[3]);
	}

	return Vector4();
}

inline Vector4 & Matrix4x4::operator[](unsigned int i) {
	switch (i) {
		case 0:  return m0;
		case 1:  return m1;
		case 2:  return m2;
		case 3:  return m3;
		default: return m3;
	}
}

inline Vector4 const & Matrix4x4::operator[](unsigned int i) const {
	switch (i) {
		case 0:  return m0;
		case 1:  return m1;
		case 2:  return m2;
		case 3:  return m3;
		default: return m3;
	}
}

FORCEINLINE Matrix4x4 Matrix4x4::operator*(const Matrix4x4 & Other) const {
	Matrix4x4 Result = Matrix4x4();

	Vector4 Col0 = Column(0), Col1 = Column(1), Col2 = Column(2), Col3 = Column(3);

	Result.m0[0] = Other.m0.Dot(Col0);
	Result.m1[0] = Other.m1.Dot(Col0);
	Result.m2[0] = Other.m2.Dot(Col0);
	Result.m3[0] = Other.m3.Dot(Col0);

	Result.m0[1] = Other.m0.Dot(Col1);
	Result.m1[1] = Other.m1.Dot(Col1);
	Result.m2[1] = Other.m2.Dot(Col1);
	Result.m3[1] = Other.m3.Dot(Col1);
				   
	Result.m0[2] = Other.m0.Dot(Col2);
	Result.m1[2] = Other.m1.Dot(Col2);
	Result.m2[2] = Other.m2.Dot(Col2);
	Result.m3[2] = Other.m3.Dot(Col2);
				   
	Result.m0[3] = Other.m0.Dot(Col3);
	Result.m1[3] = Other.m1.Dot(Col3);
	Result.m2[3] = Other.m2.Dot(Col3);
	Result.m3[3] = Other.m3.Dot(Col3);

	return Result;
}

FORCEINLINE Vector4 Matrix4x4::operator*(const Vector4 & Vector) const {
	Vector4 Result(
		Row(0).Dot(Vector),
		Row(1).Dot(Vector),
		Row(2).Dot(Vector),
		Row(3).Dot(Vector)
	);

	return Result;
}

FORCEINLINE Vector3 Matrix4x4::operator*(const Vector3 & Vector) const {
	Vector3 Result(
		Row(0).Dot(Vector),
		Row(1).Dot(Vector),
		Row(2).Dot(Vector)
	);

	return Result;
}

FORCEINLINE Matrix4x4 Matrix4x4::operator*(const float & Value) const {
	Matrix4x4 Result(*this);

	Result.m0 *= Value;
	Result.m1 *= Value;
	Result.m2 *= Value;
	Result.m3 *= Value;
	
	return Result;
}

FORCEINLINE Matrix4x4 Matrix4x4::operator/(const float & Value) const {
	Matrix4x4 Result(*this);

	Result.m0 /= Value;
	Result.m1 /= Value;
	Result.m2 /= Value;
	Result.m3 /= Value;

	return Result;
}

inline const float * Matrix4x4::PointerToValue(void) const {
	return &m0[0];
}
