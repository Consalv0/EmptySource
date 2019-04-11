#pragma once

#include <math.h>
#include <stdexcept>

#include "Vector3.h"
#include "Vector4.h"
#include "Matrix4x4.h"
#include "Quaternion.h"

FORCEINLINE Quaternion::Quaternion()
	:w(1), x(0), y(0), z(0)
{ }

FORCEINLINE Quaternion::Quaternion(Quaternion const & Other) 
	:w(Other.w), x(Other.x), y(Other.y), z(Other.z)
{ }

FORCEINLINE Quaternion::Quaternion(float const & Scale, Vector3 const & Vector)
	:w(Scale), x(Vector.x), y(Vector.y), z(Vector.z)
{ }

FORCEINLINE Quaternion::Quaternion(float const& w, float const& x, float const& y, float const& z) 
	:w(w), x(x), y(y), z(z)
{ }

inline Quaternion Quaternion::EulerAngles(Vector3 const & EulerAngles) {
	float Scale = MathConstants::DegreeToRad * 0.5F;
	Vector3 Vcos = { std::cos(EulerAngles.x * Scale), std::cos(EulerAngles.y * Scale), std::cos(EulerAngles.z * Scale) };
	Vector3 Vsin = { std::sin(EulerAngles.x * Scale), std::sin(EulerAngles.y * Scale), std::sin(EulerAngles.z * Scale) };

	return Quaternion (
		Vcos.x * Vcos.y * Vcos.z + Vsin.x * Vsin.y * Vsin.z,
		Vsin.x * Vcos.y * Vcos.z - Vcos.x * Vsin.y * Vsin.z,
		Vcos.x * Vsin.y * Vcos.z + Vsin.x * Vcos.y * Vsin.z,
		Vcos.x * Vcos.y * Vsin.z - Vsin.x * Vsin.y * Vcos.z
	);
}

FORCEINLINE Quaternion Quaternion::VectorAngle(Vector3 const & u, Vector3 const & v) {
	float NormUV = sqrt(u.Dot(u) * v.Dot(v));
	float RealPart = NormUV + u.Dot(v);
	Vector3 ImgPart;

	if (RealPart < 1.e-6f * NormUV) {
		// --- If u and v are exactly opposite, rotate 180 degrees
		// --- around an arbitrary orthogonal axis. Axis normalisation
		// --- can happen later, when we normalise the quaternion.
		RealPart = 0.F;
		ImgPart = abs(u.x) > abs(u.z) ? Vector3(-u.y, u.x) : Vector3(0.F, -u.z, u.y);
	} else {
		// --- Otherwise, build quaternion the standard way.
		ImgPart = u.Cross(v);
	}

	return Quaternion(RealPart, ImgPart.x, ImgPart.y, ImgPart.z).Normalized();
}

inline Quaternion Quaternion::AxisAngle(Vector3 const & Axis, float const & Radians) {
	float Sine = sinf(Radians * .5F);

	return Quaternion (	
		cosf(Radians * .5F),
		Axis.x * Sine,
		Axis.y * Sine,
		Axis.z * Sine
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
	} else {
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

inline float Quaternion::GetScalar() const {
	return w;
}

inline Vector3 Quaternion::GetVector() const {
	return Vector3(x, y, z);
}

inline Vector3 Quaternion::ToEulerAngles() const {
	// --- Reference 
	// --- http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/

	Vector3 EulerAngles;

	// --- Is close to the pole?
	float SingularityTest = x * y + z * w;
	if (SingularityTest > 0.499F) {
		EulerAngles[Yaw]   = 2 * atan2(x, w);
		EulerAngles[Roll]  = MathConstants::HalfPi;
		EulerAngles[Pitch] = 0;
	}
	else if (SingularityTest < -0.499F) {
		EulerAngles[Yaw]   = -2 * atan2(x, w);
		EulerAngles[Roll]  = -MathConstants::HalfPi;
		EulerAngles[Pitch] = 0;
	}
	else {
		const float sqx = x * x;
		const float sqy = y * y;
		const float sqz = z * z;
		EulerAngles[Yaw] = atan2((2 * y * w) - (2 * x * z), 1 - (2 * sqy) - (2 * sqz));
		EulerAngles[Roll] = asin(2 * SingularityTest);
		EulerAngles[Pitch] = atan2((2 * x * w) - (2 * y * z), 1 - (2 * sqx) - (2 * sqz));
	}
	return EulerAngles * MathConstants::RadToDegree;
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
	// float num = x * 2.F;
	// float num2 = y * 2.f;
	// float num3 = z * 2.f;
	// float num4 = x * num;
	// float num5 = y * num2;
	// float num6 = z * num3;
	// float num7 = x * num2;
	// float num8 = x * num3;
	// float num9 = y * num3;
	// float num10 = w * num;
	// float num11 = w * num2;
	// float num12 = w * num3;
	// return Vector3 (
	// 	(1.F - (num5 + num6)) * Vector.x + (num7 - num12) * Vector.y + (num8 + num11) * Vector.z,
	// 	(num7 + num12) * Vector.x + (1.F - (num4 + num6)) * Vector.y + (num9 - num10) * Vector.z,
	// 	(num8 - num11) * Vector.x + (num9 + num10) * Vector.y + (1.F - (num4 + num5)) * Vector.z
	// );

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
