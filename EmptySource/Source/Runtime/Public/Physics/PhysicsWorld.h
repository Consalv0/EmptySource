#pragma once

#include "Physics/Physics.h"

namespace ESource {

	class PhysicsWorld {
	public:
		PhysicsWorld();

		~PhysicsWorld();

		void SuscribePhysicsBody(class CPhysicBody * ComponentBody);

		void UnsuscribePhysicsBody(class CPhysicBody * ComponentBody);

		void RayCast(const Ray& CastedRay, TArray<RayHit> & OutHits);

		void AABBIntersection(const BoundingBox3D& AABB, TArray<CPhysicBody *> & Intersections);

	protected:
		TArray<class CPhysicBody *> PhysicsBodyArray;
	};

}