
#include <math.h>
#include <stdexcept>

#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/Quaternion.h"
#include "Math/Matrix4x4.h"

namespace ESource {

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

	inline Matrix4x4::Matrix4x4(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {

		m0[0] = m00; m0[1] = m01; m0[2] = m02; m0[3] = m03;
		m1[0] = m10; m1[1] = m11; m1[2] = m12; m1[3] = m13;
		m2[0] = m20; m2[1] = m21; m2[2] = m22; m2[3] = m23;
		m3[0] = m30; m3[1] = m31; m3[2] = m32; m3[3] = m33;
	}

	inline Matrix4x4 Matrix4x4::Identity() {
		return Matrix4x4();
	}

	inline Matrix4x4 Matrix4x4::Perspective(const float & FOV, const float & Aspect, const float & Near, const float & Far) {
		float const TanHalfFOV = tan(FOV / 2.F);
		Matrix4x4 Result(
			1.F / (Aspect * TanHalfFOV), 0.F, 0.F, 0.F,
			0.F, 1.F / (TanHalfFOV), 0.F, 0.F,
			0.F, 0.F, -(Far + Near) / (Far - Near), -1.F,
			0.F, 0.F, -(2.F * Far * Near) / (Far - Near), 0.F
		);
		return Result;
	}

	inline Matrix4x4 Matrix4x4::Orthographic(const float & Left, const float & Right, const float & Bottom, const float & Top) {
		Matrix4x4 Result(
			2.F / (Right - Left), 0.F, 0.F, 0.F,
			0.F, 2.F / (Top - Bottom), 0.F, 0.F,
			0.F, 0.F, 1.F, 0.F,
			-(Right + Left) / (Right - Left), -(Top + Bottom) / (Top - Bottom), 0.F, 1.F
		);
		return Result;
	}

	inline Matrix4x4 Matrix4x4::Orthographic(const float & Left, const float & Right, const float & Bottom, const float & Top, const float & Near, const float & Far) {
		Matrix4x4 Result(
			2.F / (Right - Left), 0.F, 0.F, 0.F,
			0.F, 2.F / (Top - Bottom), 0.F, 0.F,
			0.F, 0.F, 1.F / (Near - Far), 0.F,
			-(Right + Left) / (Right - Left), -(Top + Bottom) / (Top - Bottom), Near / (Near - Far), 1.F
		);
		return Result;
	}

	inline Matrix4x4 Matrix4x4::LookAt(const Vector3 & Eye, const Vector3 & Target, const Vector3 & Up) {
		Vector3 const Forward((Eye - Target).Normalized());
		Vector3 const Side(Forward.Cross(Up).Normalized());
		Vector3 const Upper(Side.Cross(Forward));

		return Matrix4x4(
			Side.x, Side.y, Side.z, 0.F,
			Upper.x, Upper.y, Upper.z, 0.F,
			-Forward.x, -Forward.y, -Forward.z, 0.F,
			Eye.x, Eye.y, Eye.z, 1.F
		);
	}

	inline Matrix4x4 Matrix4x4::Translation(const Vector3 & Vector) {
		return Matrix4x4(
			1.F, 0.F, 0.F, 0.F,
			0.F, 1.F, 0.F, 0.F,
			0.F, 0.F, 1.F, 0.F,
			Vector.x, Vector.y, Vector.z, 1.F
		);
	}

	inline Matrix4x4 Matrix4x4::Scaling(const Vector3 & Vector) {
		return Matrix4x4(
			Vector.x, 0.F, 0.F, 0.F,
			0.F, Vector.y, 0.F, 0.F,
			0.F, 0.F, Vector.z, 0.F,
			0.F, 0.F, 0.F, 1.F
		);
	}

	inline Matrix4x4 Matrix4x4::Rotation(const Vector3 & Axis, const float & Angle) {
		float const cosA = cos(Angle);
		float const sinA = sin(Angle);

		Vector3 AxisN(Axis.Normalized());
		Vector3 Temp(AxisN * (1.F - cosA));

		Matrix4x4 Rotation;
		Rotation.m0[0] = cosA + Temp[0] * AxisN[0];
		Rotation.m0[1] = Temp[0] * AxisN[1] + sinA * AxisN[2];
		Rotation.m0[2] = Temp[0] * AxisN[2] - sinA * AxisN[1];

		Rotation.m1[0] = Temp[1] * AxisN[0] - sinA * AxisN[2];
		Rotation.m1[1] = cosA + Temp[1] * AxisN[1];
		Rotation.m1[2] = Temp[1] * AxisN[2] + sinA * AxisN[0];

		Rotation.m2[0] = Temp[2] * AxisN[0] + sinA * AxisN[1];
		Rotation.m2[1] = Temp[2] * AxisN[1] - sinA * AxisN[0];
		Rotation.m2[2] = cosA + Temp[2] * AxisN[2];

		return Rotation;
	}

	inline Matrix4x4 Matrix4x4::Rotation(const Vector3 & EulerAngles) {
		Matrix4x4 Result;
		float Sinr, Sinp, Siny, Cosr, Cosp, Cosy;

		Siny = std::sin(EulerAngles[Yaw] * MathConstants::DegreeToRad);
		Cosy = std::cos(EulerAngles[Yaw] * MathConstants::DegreeToRad);
		Sinp = std::sin(EulerAngles[Pitch] * MathConstants::DegreeToRad);
		Cosp = std::cos(EulerAngles[Pitch] * MathConstants::DegreeToRad);
		Sinr = std::sin(EulerAngles[Roll] * MathConstants::DegreeToRad);
		Cosr = std::cos(EulerAngles[Roll] * MathConstants::DegreeToRad);

		Result.m0[0] = Cosp * Cosy;
		Result.m0[1] = Cosp * Siny;
		Result.m0[2] = -Sinp;

		Result.m1[0] = Sinr * Sinp*Cosy + Cosr * -Siny;
		Result.m1[1] = Sinr * Sinp*Siny + Cosr * Cosy;
		Result.m1[2] = Sinr * Cosp;

		Result.m2[0] = (Cosr*Sinp*Cosy + -Sinr * -Siny);
		Result.m2[1] = (Cosr*Sinp*Siny + -Sinr * Cosy);
		Result.m2[2] = Cosr * Cosp;

		Result.m3[0] = 0.F;
		Result.m3[1] = 0.F;
		Result.m3[2] = 0.F;
		Result.m3[3] = 1.F;

		return Result;
	}

	inline Matrix4x4 Matrix4x4::Rotation(const Quaternion & Quat) {
		return Quat.ToMatrix4x4();
	}

	inline void Matrix4x4::Transpose() {
		Matrix4x4 Result = Matrix4x4(GetColumn(0), GetColumn(1), GetColumn(2), GetColumn(3));
		*this = Result;
	}

	inline Matrix4x4 Matrix4x4::Transposed() const {
		return Matrix4x4(GetColumn(0), GetColumn(1), GetColumn(2), GetColumn(3));
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

		Vector4 SignA(+1.F, -1.F, +1.F, -1.F);
		Vector4 SignB(-1.F, +1.F, -1.F, +1.F);

		Matrix4x4 Result(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);

		// Vector4 Row0(Inverse[0][0], Inverse[1][0], Inverse[2][0], Inverse[3][0]);

		Vector4 Dot0(GetRow(0) * Result.GetColumn(0));
		float Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

		float OneOverDeterminant = 1.F / Dot1;

		return Result * OneOverDeterminant;
	}

	inline Vector4 Matrix4x4::GetRow(const unsigned char & i) const {
		if (i > 3) return Vector4();
		return ((Vector4*)this)[i];
	}

	inline Vector4 Matrix4x4::GetColumn(const unsigned char & i) const {
		switch (i) {
			case 0: return { m0[0], m1[0], m2[0], m3[0] };
			case 1: return { m0[1], m1[1], m2[1], m3[1] };
			case 2: return { m0[2], m1[2], m2[2], m3[2] };
			case 3: return { m0[3], m1[3], m2[3], m3[3] };
		}

		return Vector4();
	}

	inline HOST_DEVICE Vector3 Matrix4x4::ExtractTranslation() const {
		return GetRow(3);
	}

	inline HOST_DEVICE Quaternion Matrix4x4::ExtractRotation() const {
		return Quaternion::LookRotation(GetRow(2), GetRow(1));
	}

	inline HOST_DEVICE Vector3 Matrix4x4::ExtractScale() const {
		Vector3 Scale(
			GetRow(0).Magnitude(),
			GetRow(1).Magnitude(),
			GetRow(2).Magnitude()
		);
		return Scale;
	}

	inline Vector4 & Matrix4x4::operator[](unsigned char i) {
		ES_CORE_ASSERT(i <= 3, "Matrix4x4 index out of bounds");
		return ((Vector4*)this)[i];
	}

	inline Vector4 const & Matrix4x4::operator[](unsigned char i) const {
		ES_CORE_ASSERT(i <= 3, "Matrix4x4 index out of bounds");
		return ((Vector4*)this)[i];
	}

	inline HOST_DEVICE Vector3 Matrix4x4::MultiplyPoint(const Vector3 & Vector) const {
		Vector3 Result = *this * Vector;
		Result += GetColumn(3);
		Result *= 1.F / GetColumn(3).Dot(Vector4(Vector, 1.F));
		return Result;
	}

	inline Vector3 Matrix4x4::MultiplyVector(const Vector3 & Vector) const {
		return *this * Vector;
	}

	FORCEINLINE Matrix4x4 Matrix4x4::operator*(const Matrix4x4 & Other) const {
		Matrix4x4 Result = Matrix4x4();

		const Vector4 Col0 = GetColumn(0), Col1 = GetColumn(1), Col2 = GetColumn(2), Col3 = GetColumn(3);

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
			GetColumn(0).Dot(Vector),
			GetColumn(1).Dot(Vector),
			GetColumn(2).Dot(Vector),
			GetColumn(3).Dot(Vector)
		);

		return Result;
	}

	FORCEINLINE Vector3 Matrix4x4::operator*(const Vector3 & Vector) const {
		Vector4 const Vect = Vector4(Vector, 0.F);
		Vector3 Result(
			GetColumn(0).Dot(Vect),
			GetColumn(1).Dot(Vect),
			GetColumn(2).Dot(Vect)
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

}
