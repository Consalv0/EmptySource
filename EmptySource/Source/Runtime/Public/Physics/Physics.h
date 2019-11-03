#pragma once

#include "Physics/Ray.h"
#include "Math/Box3D.h"

namespace ESource {

	namespace Physics {
		inline static bool RaycastAxisAlignedBox(const Ray & CastedRay, const Box3D & AABox);
		inline static bool RaycastTriangle(const Ray & CastedRay, const Point3 & A, const Point3 & B, const Point3 & C, bool DoubleSided);
		inline static bool RaycastTriangle(RayHit & Hit, const Ray & CastedRay, const Point3 & A, const Point3 & B, const Point3 & C, bool DoubleSided);
	}

	inline bool Physics::RaycastAxisAlignedBox(const Ray & CastedRay, const Box3D & AABox) {
		Vector3 InverseRay = 1.F / CastedRay.Direction;
		const Vector3 MinPoint = AABox.GetMinPoint();
		const Vector3 MaxPoint = AABox.GetMaxPoint();

		float tmin, tmax, tymin, tymax, tzmin, tzmax;

		tmin = ((InverseRay.X > 0 ? MinPoint : MaxPoint).X - CastedRay.Origin.X) * InverseRay.X;
		tmax = ((InverseRay.X < 0 ? MinPoint : MaxPoint).X - CastedRay.Origin.X) * InverseRay.X;
		tymin = ((InverseRay.Y > 0 ? MinPoint : MaxPoint).Y - CastedRay.Origin.Y) * InverseRay.Y;
		tymax = ((InverseRay.Y < 0 ? MinPoint : MaxPoint).Y - CastedRay.Origin.Y) * InverseRay.Y;

		if ((tmin > tymax) || (tymin > tmax))
			return false;
		if (tymin > tmin)
			tmin = tymin;
		if (tymax < tmax)
			tmax = tymax;

		tzmin = ((InverseRay.Z > 0 ? MinPoint : MaxPoint).Z - CastedRay.Origin.Z) * InverseRay.Z;
		tzmax = ((InverseRay.Z < 0 ? MinPoint : MaxPoint).Z - CastedRay.Origin.Z) * InverseRay.Z;

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;
		// if (tzmin > tmin)
		// 	tmin = tzmin;
		if (tzmax < tmax)
			tmax = tzmax;

		// --- Behind ray
		// if (tmin < 0)
		// --- Front ray
		if (tmax > 0)
			return true;

		return false;
	}

	bool Physics::RaycastTriangle(const Ray & CastedRay, const Point3 & A, const Point3 & B, const Point3 & C, bool DoubleSided = false) {
		const Vector3 EdgeAB = B - A;
		const Vector3 EdgeAC = C - A;
		// const Vector3 Normal = Vector3::Cross(EdgeAB, EdgeAC);
		const Vector3 Perpendicular = CastedRay.Direction.Cross(EdgeAC);
		const float Determinant = EdgeAB.Dot(Perpendicular);

		// --- The determinant is negative the triangle is backfacing
		// --- The determinant is close to 0, the ray misses the triangle
		if ((DoubleSided ? fabs(Determinant) : Determinant) < MathConstants::SmallNumber)
			return false;

		const float InverseDet = 1.F / Determinant;

		Vector3 TriangleRay = CastedRay.Origin - A;
		float u = TriangleRay.Dot(Perpendicular) * InverseDet;
		if (u < 0 || u > 1)
			return false;

		Vector3 TriangleRayPerpendicular = TriangleRay.Cross(EdgeAB);
		float v = CastedRay.Direction.Dot(TriangleRayPerpendicular) * InverseDet;
		if (v < 0 || u + v > 1)
			return false;

		const float t = EdgeAC.Dot(TriangleRayPerpendicular) * InverseDet;
		if (t < 0)
			return false;

		return true;
	}

	bool Physics::RaycastTriangle(RayHit & Hit, const Ray & CastedRay, const Point3 & A, const Point3 & B, const Point3 & C, bool DoubleSided = false) {
		Hit.bHit = false;
		const Vector3 EdgeAB = B - A;
		const Vector3 EdgeAC = C - A;
		const Vector3 Perpendicular = Vector3::Cross(CastedRay.Direction, EdgeAC);
		const float Determinant = Vector3::Dot(EdgeAB, Perpendicular);

		// --- The determinant is negative the triangle is backfacing
		// --- The determinant is close to 0, the ray misses the triangle
		if ((DoubleSided ? fabs(Determinant) : Determinant) < MathConstants::SmallNumber)
			return false;

		const float InverseDet = 1.F / Determinant;

		const Vector3 TriangleRay = CastedRay.Origin - A;
		const float u = Vector3::Dot(TriangleRay, Perpendicular) * InverseDet;
		if (u < 0 || u > 1)
			return false;

		const Vector3 TriangleRayPerpendicular = TriangleRay.Cross(EdgeAB);
		const float v = Vector3::Dot(CastedRay.Direction, TriangleRayPerpendicular) * InverseDet;
		if (v < 0 || (u + v) > 1)
			return false;

		const float t = Vector3::Dot(EdgeAC, TriangleRayPerpendicular) * InverseDet;
		if (t < 0)
			return false;

		Hit.bHit = true;
		Hit.Stamp = t;
		Hit.Normal = Vector3::Cross(EdgeAB, EdgeAC);
		Hit.BaricenterCoordinates = Vector3(1.F - u - v, u, v);

		return true;
	}

}