#pragma once

#include "Math/MathUtility.h"
#include "Math/Vector3.h"

namespace ESource {

	struct RayHit {
	public:
		bool bHit;
		float Stamp;
		Vector3 Normal;
		Vector3 BaricenterCoordinates;
		class CPhysicBody * PhysicBody;
		int TriangleIndex;

		HOST_DEVICE FORCEINLINE RayHit() {
			bHit = false;
#ifndef __CUDACC__
			Stamp = MathConstants::BigNumber;
#else
			Stamp = 3.4e+38f;
#endif
			Normal = 0;
			BaricenterCoordinates = 0;
			TriangleIndex = -1;
			PhysicBody = NULL;
		}

		HOST_DEVICE inline bool operator < (const RayHit& Other) const {
			return (Stamp < Other.Stamp);
		}
	};

	class Ray {
	public:
		Vector3 Origin;
		Vector3 Direction;

		HOST_DEVICE FORCEINLINE Ray();
		HOST_DEVICE FORCEINLINE Ray(const Vector3& Origin, const Vector3& Direction);

		HOST_DEVICE inline Vector3 GetOrigin() const { return Origin; }
		HOST_DEVICE inline Vector3 GetDirection() const { return Direction; }

		//* Get the position in given time
		HOST_DEVICE inline Vector3 PointAt(float t) const;
	};

}

#include "Physics/Ray.inl"
