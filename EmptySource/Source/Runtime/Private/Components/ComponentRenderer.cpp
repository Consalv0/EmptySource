
#include "CoreMinimal.h"
#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Mesh.h"
#include "Resources/MeshManager.h"
#include "Core/GameObject.h"
#include "Core/CoreTime.h"
#include "Core/Application.h"

#include "Components/ComponentRenderer.h"

namespace EmptySource {

	CRenderer::CRenderer(GGameObject & GameObject) : CComponent(L"Renderer", GameObject), Model(NULL) {
	}

	bool CRenderer::Initialize() {
		LOG_CORE_DEBUG(L"Renderer '{0}'[{1:d}] Initalized", GetUniqueName(), GetUniqueID());
		return true;
	}

	void CRenderer::OnDelete() {
		LOG_CORE_DEBUG(L"Renderer '{0}'[{1:d}] Destroyed", GetUniqueName(), GetUniqueID());
	}

	void CRenderer::SetMesh(MeshPtr Value)
	{
	}

	void CRenderer::SetMaterials(TArray<class Material*> Materials)
	{
	}

	void CRenderer::SetMaterialAt(unsigned int At, Material* Mat) {
	}

	void CRenderer::OnRender() {
		if (Model == NULL) return;

		if (Model->SetUpBuffers()) {
		// 	Model->BindSubdivisionVertexArray((int)fmodf(Time::GetEpochTime<Time::Second>(), (float)Model->GetMeshData().Materials.size()));
		// 
		// 	CurrentMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, &GetGameObject().Transformation.GetLocalToWorldMatrix(), Stage->GetMatrixBuffer());
		// 	Model->DrawSubdivisionInstanciated(1, (int)fmodf(Time::GetEpochTime<Time::Second>(), (float)Model->GetMeshData().Materials.size()));
		}
	}

}