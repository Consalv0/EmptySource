
#include "CoreMinimal.h"
#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Resources/MeshManager.h"
#include "Core/GameObject.h"
#include "Core/CoreTime.h"
#include "Core/Application.h"

#include "Components/ComponentRenderable.h"

namespace EmptySource {

	CRenderable::CRenderable(GGameObject & GameObject) : CComponent(L"Renderer", GameObject), ActiveMesh() {
	}

	void CRenderable::OnDelete() {
		LOG_CORE_DEBUG(L"Renderer '{0}'[{1:d}] Destroyed", Name.GetDisplayName(), Name.GetInstanceID());
	}

	void CRenderable::SetMesh(MeshPtr Value) {
		ActiveMesh.swap(Value);
		if (ActiveMesh) {
			for (auto & MaterialLook : ActiveMesh->GetMeshData().Materials)
				Materials.try_emplace(MaterialLook.first);
			size_t MaterialSize = Materials.size();
			for (size_t i = ActiveMesh->GetMeshData().Materials.size(); i < MaterialSize; ++i)
				Materials.erase((int)i);
		}
		else
			Materials.clear();
	}

	void CRenderable::SetMaterials(TArray<MaterialPtr> & Materials) {

	}

	void CRenderable::SetMaterialAt(unsigned int At, MaterialPtr Mat) {
		Materials[At] = Mat;
	}

	const TDictionary<int, MaterialPtr> & CRenderable::GetMaterials() const {
		return Materials;
	}

	MeshPtr CRenderable::GetMesh() const {
		return ActiveMesh;
	}

	void CRenderable::OnRender() {
		if (ActiveMesh == NULL) return;

		RenderPipeline & Pipeline = Application::GetInstance()->GetRenderPipeline();
		Matrix4x4 GameObjectLWMatrix = GetGameObject().GetWorldTransform().GetLocalToWorldMatrix();
		for (auto& ItMaterial : Materials) {
			if (ItMaterial.second)
				Pipeline.SubmitMesh(ActiveMesh, ItMaterial.first, ItMaterial.second, GameObjectLWMatrix);
		}
	}

}