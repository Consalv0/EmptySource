#pragma once

#include "CoreTypes.h"
#include "Math/Matrix4x4.h"
#include "Math/Vector3.h"

namespace ESource {

	struct Plane {
	public:
		union {
			struct { float X, Y, Z, D; };
			struct { Vector3 Normal; float Distance; };
		};

		HOST_DEVICE FORCEINLINE Plane();
		HOST_DEVICE FORCEINLINE Plane(const float & X, const float & Y, const float & Z, const float & D);
		HOST_DEVICE FORCEINLINE Plane(const Vector3 & NormalizedNormal, float InD);

		HOST_DEVICE FORCEINLINE static Plane FromPointNormal(const Point3 & Point, const Vector3 & Normal);
		HOST_DEVICE FORCEINLINE static Plane From3Points(const Point3 & V0, const Point3 & V1, const Point3 & V2);

		HOST_DEVICE FORCEINLINE void Normalize();

		HOST_DEVICE FORCEINLINE float SignedDistance(const Vector3 &Pt) const;

		HOST_DEVICE FORCEINLINE static float Dot(const Plane &P, const Vector4 &V);
		HOST_DEVICE FORCEINLINE static float Dot(const Plane &P, const Vector3 &V);
		HOST_DEVICE FORCEINLINE static float DotCoord(const Plane &P, const Vector3 &V);
		HOST_DEVICE FORCEINLINE static float DotNormal(const Plane &P, const Vector3 &V);

	};

}

#include "Math/Plane.inl"