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
		:W(1), X(0), Y(0), Z(0)
	{ }

	FORCEINLINE Quaternion::Quaternion(Quaternion const & Other)
		: W(Other.W), X(Other.X), Y(Other.Y), Z(Other.Z)
	{ }

	FORCEINLINE Quaternion::Quaternion(float const & Scale, Vector3 const & Vector)
		: W(Scale), X(Vector.X), Y(Vector.Y), Z(Vector.Z)
	{ }

	FORCEINLINE Quaternion::Quaternion(float const& W, float const& X, float const& Y, float const& Z)
		: W(W), X(X), Y(Y), Z(Z)
	{ }

	inline Quaternion Quaternion::FromEulerAngles(Vector3 const & EulerAngles) {
		const float Scale = MathConstants::DegreeToRad * 0.5F;
		float HalfRoll = EulerAngles[Roll] * Scale;
		float HalfPitch = EulerAngles[Pitch] * Scale;
		float HalfYaw = EulerAngles[Yaw] * Scale;

		float SR = std::sin(HalfRoll);
		float CR = std::cos(HalfRoll);
		float SP = std::sin(HalfPitch);
		float CP = std::cos(HalfPitch);
		float SY = std::sin(HalfYaw);
		float CY = std::cos(HalfYaw);

		Quaternion EulerToQuat;
		EulerToQuat.X = (CY * SP * CR) + (SY * CP * SR);
		EulerToQuat.Y = (SY * CP * CR) - (CY * SP * SR);
		EulerToQuat.Z = (CY * CP * SR) - (SY * SP * CR);
		EulerToQuat.W = (CY * CP * CR) + (SY * SP * SR);
		return EulerToQuat;
	}

	FORCEINLINE Quaternion Quaternion::FromToRotation(Vector3 const & From, Vector3 const & To) {
		Vector3 Half = From + To;
		Half.Normalize();

		return Quaternion(
			From.Dot(Half),
			From.Y * Half.Z - From.Z * Half.Y,
			From.Z * Half.X - From.X * Half.Z,
			From.X * Half.Y - From.Y * Half.X
		).Normalized();
	}

	inline Quaternion Quaternion::FromAxisAngle(Vector3 const & Axis, float const & Radians) {
		float Sine = sinf(Radians * .5F);

		return Quaternion(
			cosf(Radians * .5F),
			Axis.X * Sine,
			Axis.Y * Sine,
			Axis.Z * Sine
		);
	}

	inline Quaternion Quaternion::FromLookRotation(Vector3 const & Forward, Vector3 const & Up) {
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

	inline void Quaternion::Interpolate(Quaternion & Out, const Quaternion & Start, const Quaternion & End, float Factor) {
		float CosTheta = Start.X * End.X + Start.Y * End.Y + Start.Z * End.Z + Start.W * End.W;

		// Adjust signs (if necessary)
		Quaternion AdjEnd = End;
		if (CosTheta < 0.F) {
			CosTheta = -CosTheta;
			AdjEnd.X = -AdjEnd.X;   // Reverse all signs
			AdjEnd.Y = -AdjEnd.Y;
			AdjEnd.Z = -AdjEnd.Z;
			AdjEnd.W = -AdjEnd.W;
		}

		// Calculate coefficients
		float sclp, sclq;
		if ((1.F - CosTheta) > 0.0001F) {
			// Standard case (slerp)
			float omega = std::acos(CosTheta); // extract theta from dot product's cos theta
			float sinom = std::sin(omega);
			sclp = std::sin((1.F - Factor) * omega) / sinom;
			sclq = std::sin(Factor * omega) / sinom;
		}
		else {
			// Very close, do linear interp (because it's faster)
			sclp = 1.F - Factor;
			sclq = Factor;
		}

		Out.X = sclp * Start.X + sclq * AdjEnd.X;
		Out.Y = sclp * Start.Y + sclq * AdjEnd.Y;
		Out.Z = sclp * Start.Z + sclq * AdjEnd.Z;
		Out.W = sclp * Start.W + sclq * AdjEnd.W;
	}

	inline float Quaternion::Magnitude() const {
		return sqrtf(X * X + Y * Y + Z * Z + W * W);
	}

	inline float Quaternion::MagnitudeSquared() const {
		return X * X + Y * Y + Z * Z + W * W;
	}

	inline void Quaternion::Normalize() {
		if (MagnitudeSquared() == 0) {
			W = 1; X = 0; Y = 0; Z = 0;
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
		float xx(X * X);
		float yy(Y * Y);
		float zz(Z * Z);
		float xz(X * Z);
		float xy(X * Y);
		float yz(Y * Z);
		float wx(W * X);
		float wy(W * Y);
		float wz(W * Z);

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
		const float SingularityTest = X * Y + Z * W;
		if (Math::Abs(SingularityTest) > 0.499995F) {
			return 0.F;
		}
		else {
			Pitch = Math::Atan2((2.F * X * W) - (2.F * Y * Z), 1.F - (2.F * Math::Square(X)) - (2.F * Math::Square(Z)));
		}
		return Pitch * MathConstants::RadToDegree;
	}

	inline float Quaternion::GetYaw() const {
		float Yaw;
		const float SingularityTest = X * Y + Z * W;
		if (SingularityTest > 0.499995F) {
			Yaw = 2.F * Math::Atan2(X, W);
		}
		else if (SingularityTest < -0.49999F) {
			Yaw = -2.F * Math::Atan2(X, W);
		}
		else {
			const float sqy = Y * Y;
			const float sqz = Z * Z;
			Yaw = Math::Atan2((2.F * Y * W) - (2.F * X * Z), 1.F - (2.F * sqy) - (2.F * sqz));
		}
		return Yaw * MathConstants::RadToDegree;
	}

	inline float Quaternion::GetRoll() const {
		float Roll;
		const float SingularityTest = X * Y + Z * W;
		if (SingularityTest > 0.499995F) {
			Roll = MathConstants::HalfPi;
		}
		else if (SingularityTest < -0.499995F) {
			Roll = -MathConstants::HalfPi;
		}
		else {
			Roll = asin(2.F * SingularityTest);
		}
		return Roll * MathConstants::RadToDegree;
	}

	inline float Quaternion::GetScalar() const {
		return W;
	}

	inline Vector3 Quaternion::GetVector() const {
		return Vector3(X, Y, Z);
	}

	inline Vector3 Quaternion::ToEulerAngles() const {
		Vector3 EulerFromQuat;

		float PitchY = std::asin(2.F * (X * W - Y * Z));
		float Test = std::cos(PitchY);
		if (Test > MathConstants::TendencyZero) {
			EulerFromQuat[Roll]  = Math::Atan2(2.F * (X * Y + Z * W), 1.F - (2.F * (Math::Square(Z) + Math::Square(X)))) * MathConstants::RadToDegree;
			EulerFromQuat[Pitch] = PitchY * MathConstants::RadToDegree;
			EulerFromQuat[Yaw]   = Math::Atan2(2.F * (Z * X + Y * W), 1.F - (2.F * (Math::Square(Y) + Math::Square(X)))) * MathConstants::RadToDegree;
		}
		else {
			EulerFromQuat[Roll]  = Math::Atan2(-2.F * (X * Y - Z * W), 1.F - (2.F * (Math::Square(Y) + Math::Square(Z)))) * MathConstants::RadToDegree;
			EulerFromQuat[Pitch] = PitchY * MathConstants::RadToDegree;
			EulerFromQuat[Yaw]   = 0.F;
		}

		if (std::isinf(EulerFromQuat[Roll])  || std::isnan(EulerFromQuat[Roll]))   EulerFromQuat[Roll] = 0.F;
		if (std::isinf(EulerFromQuat[Pitch]) || std::isnan(EulerFromQuat[Pitch])) EulerFromQuat[Pitch] = 0.F;
		if (std::isinf(EulerFromQuat[Yaw])   || std::isnan(EulerFromQuat[Yaw]))     EulerFromQuat[Yaw] = 0.F;

		return EulerFromQuat;
	}

	FORCEINLINE float Quaternion::Dot(const Quaternion & Other) const {
		return X * Other.X + Y * Other.Y + Z * Other.Z + W * Other.W;
	}

	inline Quaternion Quaternion::Cross(const Quaternion & Other) const {
		return Quaternion(
			W * Other.W - X * Other.X - Y * Other.Y - Z * Other.Z,
			W * Other.X + X * Other.W + Y * Other.Z - Z * Other.Y,
			W * Other.Y + Y * Other.W + Z * Other.X - X * Other.Z,
			W * Other.Z + Z * Other.W + X * Other.Y - Y * Other.X
		);
	}

	inline const float * Quaternion::PointerToValue() const {
		return &W;
	}

	inline float & Quaternion::operator[](unsigned char i) {
		return (&W)[i];
	}

	inline float const & Quaternion::operator[](unsigned char i) const {
		return (&W)[i];
	}

	FORCEINLINE bool Quaternion::operator==(const Quaternion& Other) const {
		return (X == Other.X && Y == Other.Y && Z == Other.Z && W == Other.W);
	}

	FORCEINLINE bool Quaternion::operator!=(const Quaternion& Other) const {
		return (X != Other.X || Y != Other.Y || Z != Other.Z || W != Other.W);
	}

	FORCEINLINE Quaternion Quaternion::operator-(void) const {
		return Quaternion(-W, -X, -Y, -Z);
	}

	FORCEINLINE Quaternion Quaternion::operator*(const float& Value) const {
		return Quaternion(W * Value, X * Value, Y * Value, Z * Value);
	}

	FORCEINLINE Quaternion Quaternion::operator/(const float& Value) const {
		if (Value == 0.F) Quaternion();
		return Quaternion(W / Value, X / Value, Y / Value, Z / Value);
	}

	FORCEINLINE Quaternion Quaternion::operator*(const Quaternion & Other) const {
		Quaternion Result;

		Result.X = W * Other.X + X * Other.W + Y * Other.Z - Z * Other.Y;
		Result.Y = W * Other.Y + Y * Other.W + Z * Other.X - X * Other.Z;
		Result.Z = W * Other.Z + Z * Other.W + X * Other.Y - Y * Other.X;
		Result.W = W * Other.W - X * Other.X - Y * Other.Y - Z * Other.Z;

		return Result;
	}

	inline Vector3 Quaternion::operator*(const Vector3 & Vector) const {
		Vector3 const QuatVector(GetVector());
		Vector3 const QV(Vector3::Cross(QuatVector, Vector));
		Vector3 const QQV(Vector3::Cross(QuatVector, QV));

		return Vector + ((QV * W) + QQV) * 2.F;
	}

	FORCEINLINE Quaternion & Quaternion::operator*=(const Quaternion & Other) {
		*this = *this * Other;
		return *this;
	}

	FORCEINLINE Quaternion& Quaternion::operator*=(const float& Value) {
		W *= Value;
		X *= Value;
		Y *= Value;
		Z *= Value;
		return *this;
	}

	FORCEINLINE Quaternion& Quaternion::operator/=(const float& Value) {
		if (Value == 0.F) W = X = Y = Z = 0;
		W /= Value;
		X /= Value;
		Y /= Value;
		Z /= Value;
		return *this;
	}

}