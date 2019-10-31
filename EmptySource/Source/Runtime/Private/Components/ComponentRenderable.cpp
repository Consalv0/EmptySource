
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

	CRenderable::CRenderable(GGameObject & GameObject) : CComponent(L"Rendererable", GameObject), ActiveMesh() {
	}

	void CRenderable::OnDelete() {
	}

	void CRenderable::SetMesh(RMeshPtr Value) {
		ActiveMesh.swap(Value);
		if (ActiveMesh != NULL) {
			for (auto & MaterialLook : ActiveMesh->GetVertexData().MaterialsMap)
				Materials.try_emplace(MaterialLook.first);
			size_t MaterialSize = Materials.size();
			for (size_t i = ActiveMesh->GetVertexData().MaterialsMap.size(); i < MaterialSize; ++i)
				Materials.erase((int)i);
		}
		else
			Materials.clear();
	}

	void CRenderable::SetMaterials(TArray<MaterialPtr> & Materials) {

	}

	void CRenderable::SetMaterialAt(uint32_t At, MaterialPtr Mat) {
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
		Matrix4x4 GameObjectLWMatrix = GetGameObject().GetWorldMatrix();
		for (auto& ItMaterial : Materials) {
			if (ItMaterial.second)
				Pipeline.SubmitSubmesh(ActiveMesh, ItMaterial.first, ItMaterial.second, GameObjectLWMatrix);
		}
	}

}