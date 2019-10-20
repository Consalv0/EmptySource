#pragma once

#include <math.h>
#include <stdexcept>

#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/Matrix4x4.h"
#include "Math/Matrix3x3.h"
#include "Math/Quaternion.h"

namespace ESource {

	FORCEINLINE Quaternion::Quaternion()
		:w(1), x(0), y(0), z(0)
	{ }

	FORCEINLINE Quaternion::Quaternion(Quaternion const & Other)
		: w(Other.w), x(Other.x), y(Other.y), z(Other.z)
	{ }

	FORCEINLINE Quaternion::Quaternion(float const & Scale, Vector3 const & Vector)
		: w(Scale), x(Vector.x), y(Vector.y), z(Vector.z)
	{ }

	FORCEINLINE Quaternion::Quaternion(float const& w, float const& x, float const& y, float const& z)
		: w(w), x(x), y(y), z(z)
	{ }

	inline Quaternion Quaternion::EulerAngles(Vector3 const & EulerAngles) {
		float Scale = MathConstants::DegreeToRad * 0.5F;
		float HalfRoll  =  EulerAngles[Roll] * Scale;
		float HalfPitch = EulerAngles[Pitch] * Scale;
		float HalfYaw   =   EulerAngles[Yaw] * Scale;

		float SinRoll  = std::sin(HalfRoll);
		float CosRoll  = std::cos(HalfRoll);
		float SinPitch = std::sin(HalfPitch);
		float CosPitch = std::cos(HalfPitch);
		float SinYaw   = std::sin(HalfYaw);
		float CosYaw   = std::cos(HalfYaw);

		float CosYawPitch = CosYaw * CosPitch;
		float SinYawPitch = SinYaw * SinPitch;

		Quaternion Result;
		Result.x = (CosYaw * SinPitch * CosRoll) + (SinYaw * CosPitch * SinRoll);
		Result.y = (SinYaw * CosPitch * CosRoll) - (CosYaw * SinPitch * SinRoll);
		Result.z = (CosYawPitch * SinRoll) - (SinYawPitch * CosRoll);
		Result.w = (CosYawPitch * CosRoll) + (SinYawPitch * SinRoll);
		return Result;
	}

	FORCEINLINE Quaternion Quaternion::FromToRotation(Vector3 const & From, Vector3 const & To) {
		Vector3 Half = From + To;
		Half.Normalize();

		return Quaternion(
			From.Dot(Half),
			From.y * Half.z - From.z * Half.y,
			From.z * Half.x - From.x * Half.z,
			From.x * Half.y - From.y * Half.x
		).Normalized();
	}

	inline Quaternion Quaternion::AxisAngle(Vector3 const & Axis, float const & Radians) {
		float Sine = sinf(Radians * .5F);

		return Quaternion(
			cosf(Radians * .5F),
			Axis.x * Sine,
			Axis.y * Sine,
			Axis.z * Sine
		);
	}

	inline Quaternion Quaternion::LookRotation(Vector3 const & Forward, Vector3 const & Up) {
		const Vector3 Normal = Forward.Normalized();
		const Vector3 Tangent = Vector3::Cross(Up == Normal ? Up + 0.001F : Up, Normal).Normalized();
		const Vector3 Bitangent = Vector3::Cross(Normal, Tangent);

		Matrix3x3 LookSpace(
			Tangent, Bitangent, Normal
		);

		return Quaternion::FromMatrix(LookSpace);
	}

	inline Quaternion Quaternion::FromMatrix(Matrix3x3 const & Matrix) {
		const float tr = Matrix.m0[0] + Matrix.m1[1] + Matrix.m2[2];

		if (tr > 0.F) {
			float num0 = sqrtf(tr + 1.F);
			float num1 = 0.5f / num0;
			return Quaternion(
				num0 * 0.5f,
				(Matrix.m1[2] - Matrix.m2[1]) * num1,
				(Matrix.m2[0] - Matrix.m0[2]) * num1,
				(Matrix.m0[1] - Matrix.m1[0]) * num1
			);
		}
		if ((Matrix.m0[0] >= Matrix.m1[1]) && (Matrix.m0[0] >= Matrix.m2[2])) {
			float num7 = sqrtf(((1.F + Matrix.m0[0]) - Matrix.m1[1]) - Matrix.m2[2]);
			float num4 = 0.5f / num7;
			return Quaternion(
				(Matrix.m1[2] - Matrix.m2[1]) * num4,
				0.5f * num7,
				(Matrix.m0[1] + Matrix.m1[0]) * num4,
				(Matrix.m0[2] + Matrix.m2[0]) * num4
			);
		}
		if (Matrix.m1[1] > Matrix.m2[2]) {
			float num6 = sqrtf(((1.F + Matrix.m1[1]) - Matrix.m0[0]) - Matrix.m2[2]);
			float num3 = 0.5f / num6;
			return Quaternion(
				(Matrix.m2[0] - Matrix.m0[2]) * num3,
				(Matrix.m1[0] + Matrix.m0[1]) * num3,
				0.5f * num6,
				(Matrix.m2[1] + Matrix.m1[2]) * num3
			);
		}

		float num5 = sqrtf(((1.F + Matrix.m2[2]) - Matrix.m0[0]) - Matrix.m1[1]);
		float num2 = 0.5F / num5;
		return Quaternion(
			(Matrix.m0[1] - Matrix.m1[0]) * num2,
			(Matrix.m2[0] + Matrix.m0[2]) * num2,
			(Matrix.m2[1] + Matrix.m1[2]) * num2,
			0.5F * num5
		);
	}

	inline float Quaternion::Magnitude() const {
		return sqrtf(x * x + y * y + z * z + w * w);
	}

	inline float Quaternion::MagnitudeSquared() const {
		return x * x + y * y + z * z + w * w;
	}

	inline void Quaternion::Normalize() {
		if (MagnitudeSquared() == 0) {
			w = 1; x = 0; y = 0; z = 0;
		}
		else {
			*this /= Magnitude();
		}
	}

	inline Quaternion Quaternion::Normalized() const {
		if (MagnitudeSquared() == 0) return Quaternion();
		Quaternion Result = Quaternion(*this);
		return Result /= Magnitude();
	}

	inline Quaternion Quaternion::Conjugated() const {
		return Quaternion(GetScalar(), GetVector() * -1.F);
	}

	inline Quaternion Quaternion::Inversed() const {
		float AbsoluteValue = Magnitude();
		AbsoluteValue *= AbsoluteValue;
		AbsoluteValue = 1 / AbsoluteValue;

		Quaternion ConjugateVal = Conjugated();

		float Scalar = ConjugateVal.GetScalar() * AbsoluteValue;
		Vector3 Imaginary = ConjugateVal.GetVector() * AbsoluteValue;

		return Quaternion(Scalar, Imaginary);
	}

	inline Matrix4x4 Quaternion::ToMatrix4x4() const {
		Matrix4x4 Result;
		float xx(x * x);
		float yy(y * y);
		float zz(z * z);
		float xz(x * z);
		float xy(x * y);
		float yz(y * z);
		float wx(w * x);
		float wy(w * y);
		float wz(w * z);

		Result[0][0] = 1.F - 2.F * (yy + zz);
		Result[0][1] = 2.F * (xy + wz);
		Result[0][2] = 2.F * (xz - wy);

		Result[1][0] = 2.F * (xy - wz);
		Result[1][1] = 1.F - 2.f * (xx + zz);
		Result[1][2] = 2.F * (yz + wx);

		Result[2][0] = 2.F * (xz + wy);
		Result[2][1] = 2.F * (yz - wx);
		Result[2][2] = 1.F - 2.F * (xx + yy);
		return Result;
	}

	inline float Quaternion::GetPitch() const {
		float Pitch;
		const float SingularityTest = x * y + z * w;
		if (std::abs(SingularityTest) > 0.499F) {
			return 0.F;
		}
		else {
			const float sqx = x * x;
			const float sqz = z * z;
			Pitch = atan2((2 * x * w) - (2 * y * z), 1 - (2 * sqx) - (2 * sqz));
		}
		return Pitch * MathConstants::RadToDegree;
	}

	inline float Quaternion::GetYaw() const {
		float Yaw;
		const float SingularityTest = x * y + z * w;
		if (SingularityTest > 0.499F) {
			Yaw = 2 * atan2(x, w);
		}
		else if (SingularityTest < -0.499F) {
			Yaw = -2 * atan2(x, w);
		}
		else {
			const float sqy = y * y;
			const float sqz = z * z;
			Yaw = atan2((2 * y * w) - (2 * x * z), 1 - (2 * sqy) - (2 * sqz));
		}
		return Yaw * MathConstants::RadToDegree;
	}

	inline float Quaternion::GetRoll() const {
		float Roll;
		const float SingularityTest = x * y + z * w;
		if (SingularityTest > 0.499F) {
			Roll = MathConstants::HalfPi;
		}
		else if (SingularityTest < -0.499F) {
			Roll = -MathConstants::HalfPi;
		}
		else {
			Roll = asin(2 * SingularityTest);
		}
		return Roll * MathConstants::RadToDegree;
	}

	inline float Quaternion::GetScalar() const {
		return w;
	}

	inline Vector3 Quaternion::GetVector() const {
		return Vector3(x, y, z);
	}

	inline Vector3 Quaternion::ToEulerAngles() const {
		float xx(x * x);
		float yy(y * y);
		float zz(z * z);
		float xy(x * y);
		float zw(z * w);
		float zx(z * x);
		float yw(y * w);
		float yz(y * z);
		float xw(x * w);

		Vector3 Result;
		Result[Pitch] = std::asin(2.F * (xw - yz));
		double Test = std::cos((double)Result[Pitch]);
		if (Test > MathConstants::TendencyZero) {
			Result[Roll] = std::atan2(2.F * (xy + zw), 1.F - (2.F * (zz + xx)));
			Result[Yaw]  = std::atan2(2.F * (zx + yw), 1.F - (2.F * (yy + xx)));
		} else {
			Result[Roll] = std::atan2(-2.F * (xy - zw), 1.F - (2.F * (yy + zz)));
			Result[Yaw] = 0.F;
		}

		if (std::isinf(Result[Roll])  || std::isnan(Result[Roll]))   Result[Roll] = 0.F;
		if (std::isinf(Result[Pitch]) || std::isnan(Result[Pitch])) Result[Pitch] = 0.F;
		if (std::isinf(Result[Yaw])   || std::isnan(Result[Yaw]))     Result[Yaw] = 0.F;
		return Result * MathConstants::RadToDegree;
	}

	FORCEINLINE float Quaternion::Dot(const Quaternion & Other) const {
		return x * Other.x + y * Other.y + z * Other.z + w * Other.w;
	}

	inline Quaternion Quaternion::Cross(const Quaternion & Other) const {
		return Quaternion(
			w * Other.w - x * Other.x - y * Other.y - z * Other.z,
			w * Other.x + x * Other.w + y * Other.z - z * Other.y,
			w * Other.y + y * Other.w + z * Other.x - x * Other.z,
			w * Other.z + z * Other.w + x * Other.y - y * Other.x
		);
	}

	inline const float * Quaternion::PointerToValue() const {
		return &w;
	}

	inline float & Quaternion::operator[](unsigned int i) {
		return (&w)[i];
	}

	inline float const & Quaternion::operator[](unsigned int i) const {
		return (&w)[i];
	}

	FORCEINLINE bool Quaternion::operator==(const Quaternion& Other) const {
		return (x == Other.x && y == Other.y && z == Other.z && w == Other.w);
	}

	FORCEINLINE bool Quaternion::operator!=(const Quaternion& Other) const {
		return (x != Other.x || y != Other.y || z != Other.z || w != Other.w);
	}

	FORCEINLINE Quaternion Quaternion::operator-(void) const {
		return Quaternion(-w, -x, -y, -z);
	}

	FORCEINLINE Quaternion Quaternion::operator*(const float& Value) const {
		return Quaternion(w * Value, x * Value, y * Value, z * Value);
	}

	FORCEINLINE Quaternion Quaternion::operator/(const float& Value) const {
		if (Value == 0.F) Quaternion();
		return Quaternion(w / Value, x / Value, y / Value, z / Value);
	}

	FORCEINLINE Quaternion Quaternion::operator*(const Quaternion & Other) const {
		Quaternion Result;

		Result.x = w * Other.x + x * Other.w + y * Other.z - z * Other.y;
		Result.y = w * Other.y + y * Other.w + z * Other.x - x * Other.z;
		Result.z = w * Other.z + z * Other.w + x * Other.y - y * Other.x;
		Result.w = w * Other.w - x * Other.x - y * Other.y - z * Other.z;

		return Result;
	}

	inline Vector3 Quaternion::operator*(const Vector3 & Vector) const {
		Vector3 const QuatVector(GetVector());
		Vector3 const QV(Vector3::Cross(QuatVector, Vector));
		Vector3 const QQV(Vector3::Cross(QuatVector, QV));

		return Vector + ((QV * w) + QQV) * 2.F;
	}

	FORCEINLINE Quaternion & Quaternion::operator*=(const Quaternion & Other) {
		*this = *this * Other;
		return *this;
	}

	FORCEINLINE Quaternion& Quaternion::operator*=(const float& Value) {
		w *= Value;
		x *= Value;
		y *= Value;
		z *= Value;
		return *this;
	}

	FORCEINLINE Quaternion& Quaternion::operator/=(const float& Value) {
		if (Value == 0.F) w = x = y = z = 0;
		w /= Value;
		x /= Value;
		y /= Value;
		z /= Value;
		return *this;
	}

}