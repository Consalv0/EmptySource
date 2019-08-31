
#include "CoreMinimal.h"
#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Mesh.h"
#include "Resources/MeshLoader.h"
#include "Core/GameObject.h"
#include "Core/CoreTime.h"
#include "Core/Application.h"

#include "Components/ComponentRenderer.h"

namespace EmptySource {

	CRenderer::CRenderer(GGameObject & GameObject) : CComponent(L"Renderer", GameObject), Model(NULL) {
	}

	bool CRenderer::Initialize() {
		RenderStage * TestStage = Application::GetInstance()->GetRenderPipeline().GetStage(L"TestStage");
		if (TestStage != NULL) {
			// TestStage->OnRenderEvent.AttachObserver(&TestStageObserver);
			TestStageObserver.AddCallback("Render", std::bind(&CRenderer::Render, this));
		}

		LOG_CORE_DEBUG(L"Renderer '{0}'[{1:d}] Initalized", GetUniqueName(), GetUniqueID());

		MeshLoader::LoadAsync(FileManager::GetFile(L"Resources/Models/EscafandraMV1971.fbx"), true, [this](MeshLoader::FileData & ModelData) {
			for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
				Model = new Mesh(&(*Data));
				Model->SetUpBuffers();
			}
		});

		return true;
	}

	void CRenderer::OnDelete() {
		TestStageObserver.RemoveAllCallbacks();
		LOG_CORE_DEBUG(L"Renderer '{0}'[{1:d}] Destroyed", GetUniqueName(), GetUniqueID());
	}

	void CRenderer::Render() {
		if (Model == NULL) return;

		RenderStage * Stage = Application::GetInstance()->GetRenderPipeline().GetStage(L"TestStage");
		if (Stage == NULL) return;

		if (Model->SetUpBuffers()) {
			Model->BindSubdivisionVertexArray((int)fmodf(Time::GetEpochTime<Time::Second>(), (float)Model->GetMeshData().Materials.size()));

			Material * CurrentMaterial = Stage->CurrentMaterial;
			CurrentMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, &GetGameObject().Transformation.GetLocalToWorldMatrix(), Stage->GetMatrixBuffer());
			Model->DrawSubdivisionInstanciated(1, (int)fmodf(Time::GetEpochTime<Time::Second>(), (float)Model->GetMeshData().Materials.size()));
		}
	}

}