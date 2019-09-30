
#include "CoreMinimal.h"
#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Resources/ModelManager.h"
#include "Core/GameObject.h"
#include "Core/CoreTime.h"
#include "Core/Application.h"

#include "Components/ComponentRenderable.h"

namespace ESource {

	CRenderable::CRenderable(GGameObject & GameObject) : CComponent(L"Renderer", GameObject), ActiveMesh() {
	}

	void CRenderable::OnDelete() {
		LOG_CORE_DEBUG(L"Renderer '{0}'[{1:d}] Destroyed", Name.GetDisplayName(), Name.GetInstanceID());
	}

	void CRenderable::SetMesh(RMeshPtr Value) {
		ActiveMesh.swap(Value);
		if (ActiveMesh != NULL) {
			for (auto & MaterialLook : ActiveMesh->GetVertexData().Materials)
				Materials.try_emplace(MaterialLook.first);
			size_t MaterialSize = Materials.size();
			for (size_t i = ActiveMesh->GetVertexData().Materials.size(); i < MaterialSize; ++i)
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

	RMeshPtr CRenderable::GetMesh() const {
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