
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Rendering/Material.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Mesh/Mesh.h"
#include "Resources/MeshLoader.h"
#include "Engine/GameObject.h"
#include "Engine/CoreTime.h"
#include "Engine/Application.h"

#include "Components/ComponentRenderer.h"

namespace EmptySource {

	CRenderer::CRenderer(GGameObject & GameObject) : CComponent(L"Renderer", GameObject), Model(NULL) {
	}

	bool CRenderer::Initialize() {
		RenderStage * TestStage = Application::GetInstance()->GetRenderPipeline().GetStage(L"TestStage");
		if (TestStage != NULL) {
			TestStage->OnRenderEvent.AttachObserver(&TestStageObserver);
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
		LOG_CORE_DEBUG(L"Renderer '%ls'[%d] Destroyed", GetUniqueName(), GetUniqueID());
	}

	void CRenderer::Render() {
		if (Model == NULL) return;

		RenderStage * Stage = Application::GetInstance()->GetRenderPipeline().GetStage(L"TestStage");
		if (Stage == NULL) return;

		if (Model->SetUpBuffers()) {
			Model->BindSubdivisionVertexArray((int)fmodf(Time::GetEpochTimeSeconds(), (float)Model->Data.Materials.size()));

			Material * CurrentMaterial = Stage->CurrentMaterial;
			CurrentMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, &GetGameObject().Transformation.GetLocalToWorldMatrix(), Stage->GetMatrixBuffer());
			Model->DrawSubdivisionInstanciated(1, (int)fmodf(Time::GetEpochTimeSeconds(), (float)Model->Data.Materials.size()));
		}
	}

}