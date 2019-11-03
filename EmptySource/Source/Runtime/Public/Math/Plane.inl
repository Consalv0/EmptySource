#pragma once

#include <math.h>

#include "Math/Plane.h"

namespace ESource {

	FORCEINLINE Plane::Plane() : X(), Y(), Z(), D() { }

	FORCEINLINE Plane::Plane(const float & X, const float & Y, const float & Z, const float & D) : X(X), Y(Y), Z(Z), D(D) { }

	FORCEINLINE Plane::Plane(const Vector3 & NormalizedNormal, float InD) {
		X = NormalizedNormal.X;
		Y = NormalizedNormal.Y;
		Z = NormalizedNormal.Z;
		D = InD;
	}

	FORCEINLINE Plane Plane::FromPointNormal(const Point3 & Point, const Vector3 & Normal) {
		Vector3 NormalizedNormal = Normal.Normalized();
		return Plane(
			NormalizedNormal.X,
			NormalizedNormal.Y,
			NormalizedNormal.Z,
			-Vector3::Dot(Point, NormalizedNormal)
		);
	}

	FORCEINLINE Plane Plane::From3Points(const Point3 & V0, const Point3 & V1, const Point3 & V2) {
		Vector3 Normal = Vector3::Cross(V1 - V0, V2 - V0);
		Normal.Normalize();
		FromPointNormal(V0, Normal);
	}

	FORCEINLINE void Plane::Normalize() {
		float Distance = sqrtf(X*X + Y*Y + Z*Z);
		X /= Distance;
		Y /= Distance;
		Z /= Distance;
		D /= Distance;
	}

	FORCEINLINE float Plane::SignedDistance(const Vector3 &Pt) const {
		return (X * Pt.X + Y * Pt.Y + Z * Pt.Z + D);
	}

	FORCEINLINE float Plane::Dot(const Plane &P, const Vector4 &V) {
		return P.X * V.X + P.Y * V.Y + P.Z * V.Z + P.D * V.W;
	}

	FORCEINLINE float Plane::Dot(const Plane &P, const Vector3 &V) {
		return P.X * V.X + P.Y * V.Y + P.Z * V.Z + P.D;
	}

	FORCEINLINE float Plane::DotCoord(const Plane &P, const Vector3 &V) {
		return P.X * V.X + P.Y * V.Y + P.Z * V.Z + P.D;
	}

	FORCEINLINE float Plane::DotNormal(const Plane &P, const Vector3 &V) {
		return P.X * V.X + P.Y * V.Y + P.Z * V.Z;
	}

}