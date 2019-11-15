
#include "CoreMinimal.h"

#include "Resources/ModelResource.h"
#include "Physics/Physics.h"
#include "Physics/PhysicsWorld.h"
#include "Components/ComponentPhysicBody.h"

#include "Core/GameObject.h"

namespace ESource {

	void PhysicsWorld::SuscribePhysicsBody(CPhysicBody * ComponentBody) {
		PhysicsBodyArray.push_back(ComponentBody);
	}

	void PhysicsWorld::UnsuscribePhysicsBody(CPhysicBody * ComponentBody) {
		for (TArray<CPhysicBody *>::const_iterator PhysicsBodyIt = PhysicsBodyArray.begin(); PhysicsBodyIt != PhysicsBodyArray.end(); ++PhysicsBodyIt) {
			if (*PhysicsBodyIt == ComponentBody) {
				PhysicsBodyArray.erase(PhysicsBodyIt);
				break;
			}
		}
	}

	void PhysicsWorld::RayCast(const Ray & CastedRay, TArray<RayHit> & Hits) {
		size_t TotalHitCount = 0;
		Hits.clear();
		for (size_t PhysicsBodyCount = 0; PhysicsBodyCount >= 0 && PhysicsBodyCount < PhysicsBodyArray.size(); ++PhysicsBodyCount) {
			CPhysicBody * PhysicsBody = PhysicsBodyArray[PhysicsBodyCount];
			Transform & BodyTransform = PhysicsBody->GetGameObject().GetWorldTransform();
			const Matrix4x4 TransformMat = BodyTransform.GetLocalToWorldMatrix();
			const Matrix4x4 InverseTransform = BodyTransform.GetWorldToLocalMatrix();

			const MeshData * ModelData = PhysicsBody->GetMeshData();
			if (ModelData == NULL) continue;

			BoundingBox3D ModelSpaceAABox = ModelData->Bounding.Transform(TransformMat);

			if (Physics::RaycastAxisAlignedBox(CastedRay, ModelSpaceAABox)) {
				RayHit Hit;
				Ray ModelSpaceCameraRay(
					InverseTransform.MultiplyPoint(CastedRay.GetOrigin()),
					InverseTransform.MultiplyVector(CastedRay.GetDirection())
				);
				for (MeshFaces::const_iterator Face = ModelData->Faces.begin(); Face != ModelData->Faces.end(); ++Face) {
					if (Physics::RaycastTriangle(
						Hit, ModelSpaceCameraRay,
						ModelData->StaticVertices[(*Face)[0]].Position,
						ModelData->StaticVertices[(*Face)[1]].Position,
						ModelData->StaticVertices[(*Face)[2]].Position, PhysicsBody->bDoubleSided
					)) {
						Hit.TriangleIndex = int(Face - ModelData->Faces.begin());
						Hit.PhysicBody = PhysicsBody;
						Hits.push_back(Hit);
					}
				}
			
				std::sort(Hits.begin(), Hits.end());
				TotalHitCount += Hits.size();
			
				if (Hits.size() > 0 && Hits[0].bHit) {
					if (Hits[0].TriangleIndex < ModelData->Faces.size()) {
						IntVector3 Face = ModelData->Faces[Hits[0].TriangleIndex];
						const Vector3 & N0 = ModelData->StaticVertices[Face[0]].Normal;
						const Vector3 & N1 = ModelData->StaticVertices[Face[1]].Normal;
						const Vector3 & N2 = ModelData->StaticVertices[Face[2]].Normal;
						Vector3 InterpolatedNormal =
							N0 * Hits[0].BaricenterCoordinates[0] +
							N1 * Hits[0].BaricenterCoordinates[1] +
							N2 * Hits[0].BaricenterCoordinates[2];

						Hits[0].Normal = TransformMat.Inversed().Transposed().MultiplyVector(InterpolatedNormal);
						Hits[0].Normal.Normalize();
					}
					else {
						Hits[0].bHit = false;
					}
				}
			}
		}
	}

	PhysicsWorld::PhysicsWorld() : PhysicsBodyArray() {
	}

	PhysicsWorld::~PhysicsWorld() {
		PhysicsBodyArray.clear();
	}
}