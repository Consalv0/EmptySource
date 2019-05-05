#pragma once

#include "../include/Math/Ray.h"
#include "../include/Math/Box3D.h"

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

	tmin = ((InverseRay.x > 0 ? MinPoint : MaxPoint).x - CastedRay.Origin.x) * InverseRay.x;
	tmax = ((InverseRay.x < 0 ? MinPoint : MaxPoint).x - CastedRay.Origin.x) * InverseRay.x;
	tymin = ((InverseRay.y > 0 ? MinPoint : MaxPoint).y - CastedRay.Origin.y) * InverseRay.y;
	tymax = ((InverseRay.y < 0 ? MinPoint : MaxPoint).y - CastedRay.Origin.y) * InverseRay.y;

	if ((tmin > tymax) || (tymin > tmax))
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = ((InverseRay.z > 0 ? MinPoint : MaxPoint).z - CastedRay.Origin.z) * InverseRay.z;
	tzmax = ((InverseRay.z < 0 ? MinPoint : MaxPoint).z - CastedRay.Origin.z) * InverseRay.z;

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
	if (v < 0 || (u + v) > 1) 
		return false;

	const float t = EdgeAC.Dot(TriangleRayPerpendicular) * InverseDet;
	if (t < 0)
		return false;

	Hit.bHit = true;
	Hit.Stamp = t;
	Hit.Normal = Vector3::Cross(EdgeAB, EdgeAC);
	Hit.BaricenterCoordinates = Vector3(1.F - u - v, u, v);

	return true;
}
