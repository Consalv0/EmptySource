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

inline Quaternion::Quaternion(Vector3 const & Axis, float const & Angle) {
	float Sine = sinf(Angle * .5F);

	w = cosf(Angle * .5F);
	x = Axis.x * Sine;
	y = Axis.y * Sine;
	z = Axis.z * Sine;
}

FORCEINLINE Quaternion::Quaternion(float const & Scale, Vector3 const & Vector)
	:w(Scale), x(Vector.x), y(Vector.y), z(Vector.z)
{ }

FORCEINLINE Quaternion::Quaternion(float const& _w, float const& _x, float const& _y, float const& _z) 
	:w(_w), x(_x), y(_y), z(_z)
{ }

FORCEINLINE Quaternion::Quaternion(Vector3 const & u, Vector3 const & v) {
	float NormUV = sqrt(u.Dot(u) * v.Dot(v));
	float RealPart = NormUV + u.Dot(v);
	Vector3 ImgPart;

	if (RealPart < 1.e-6f * NormUV) {
		// If u and v are exactly opposite, rotate 180 degrees
		// around an arbitrary orthogonal axis. Axis normalisation
		// can happen later, when we normalise the quaternion.
		RealPart = 0.F;
		ImgPart = abs(u.x) > abs(u.z) ? Vector3(-u.y, u.x) : Vector3(0.F, -u.z, u.y);
	} else {
		// Otherwise, build quaternion the standard way.
		ImgPart = u.Cross(v);
	}

	*this = Quaternion(RealPart, ImgPart.x, ImgPart.y, ImgPart.z).Normalized();
}

inline Quaternion::Quaternion(Vector3 const & Angles) {
	Vector3 Vcos = { std::cos(Angles.x * 0.5F), std::cos(Angles.y * 0.5F), std::cos(Angles.z * 0.5F) } ;
	Vector3 Vsin = { std::sin(Angles.x * 0.5F), std::sin(Angles.y * 0.5F), std::sin(Angles.z * 0.5F) };

	this->w = Vcos.x * Vcos.y * Vcos.z + Vsin.x * Vsin.y * Vsin.z;
	this->x = Vsin.x * Vcos.y * Vcos.z - Vcos.x * Vsin.y * Vsin.z;
	this->y = Vcos.x * Vsin.y * Vcos.z + Vsin.x * Vcos.y * Vsin.z;
	this->z = Vcos.x * Vcos.y * Vsin.z - Vsin.x * Vsin.y * Vcos.z;
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
	// Is close to the pole?
	const float SingularityTest = z * x - w * y;
	const float YawY = 2.F * (w * z + x * y);
	const float YawX = ( 1.F - 2.F*((y * y) + (z * z)) );

	// Reference 
	// http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/

	Vector3 EulerAngles;

	if (SingularityTest > 0.4999F) { // NortPole
		EulerAngles.x = 1.5708F;
		EulerAngles.y = atan2(YawY, YawX);
		EulerAngles.z = EulerAngles.y - ( 2.F * atan2(x, w) );
	} else if (SingularityTest < -0.4999F) { // SouthPole
		EulerAngles.x = -1.5708F;
		EulerAngles.y = atan2f(YawY, YawX);
		EulerAngles.z = -EulerAngles.y - ( 2.F * atan2f(x, w) );
	} else {
		EulerAngles.x = asin( 2.F * (SingularityTest) );
		EulerAngles.y = atan2(YawY, YawX);
		EulerAngles.z = atan2( -2.F * (w * x + y * z), ( 1.F - 2.F * ((x * x) + (y * y)) ) );
	}

	return EulerAngles;
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
	if (Value == 0) Quaternion();
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
	if (Value == 0) w = x = y = z = 0;
	w /= Value;
	x /= Value;
	y /= Value;
	z /= Value;
	return *this;
}
