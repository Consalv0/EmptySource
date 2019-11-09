
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"

#include "Resources/ModelResource.h"
#include "Physics/PhysicsWorld.h"
#include "Components/ComponentPhysicBody.h"

#include "Rendering/RenderPipeline.h"
#include "Rendering/Material.h"
#include "Rendering/MeshPrimitives.h"

namespace ESource {
	
	void CPhysicBody::SetMesh(RMeshPtr & Mesh) {
		ActiveMesh = Mesh;
	}

	MeshData * CPhysicBody::GetMeshData() {
		if (ActiveMesh && ActiveMesh->IsValid()) {
			return &ActiveMesh->GetVertexData();
		}
		return NULL;
	}

	CPhysicBody::CPhysicBody(ESource::GGameObject & GameObject)
		: CComponent(L"PhysicBody", GameObject), ActiveMesh(NULL) {
	}

	void CPhysicBody::OnRender() {
		if (ActiveMesh == NULL || !ActiveMesh->IsValid()) return;

		RenderPipeline & Pipeline = Application::GetInstance()->GetRenderPipeline();
		Matrix4x4 GameObjectLWMatrix = GetGameObject().GetWorldMatrix();

		MaterialPtr DebugMaterial = MaterialManager::GetInstance().GetMaterial(L"DebugMaterial");
		
		BoundingBox3D ModelSpaceAABox = ActiveMesh->GetVertexData().Bounding.Transform(GameObjectLWMatrix);
		RMeshPtr CubeMesh = ModelManager::GetInstance().GetMesh(IName(L"Cube", 0));

		if (CubeMesh && DebugMaterial)
			Pipeline.SubmitSubmesh(CubeMesh, 0, DebugMaterial, Matrix4x4::Translation(ModelSpaceAABox.GetCenter()) * Matrix4x4::Scaling(ModelSpaceAABox.GetSize()), 1);
	}

	void CPhysicBody::OnAttach() {
		Application::GetInstance()->GetPhysicsWorld().SuscribePhysicsBody(this);
	}

	void CPhysicBody::OnDelete() {
		Application::GetInstance()->GetPhysicsWorld().UnsuscribePhysicsBody(this);
	}

}