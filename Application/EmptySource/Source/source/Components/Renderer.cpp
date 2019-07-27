
#include "../include/Core.h"
#include "../include/Utility/LogCore.h"
#include "../include/Components/Renderer.h"
#include "../include/GameObject.h"
#include "../include/Material.h"
#include "../include/RenderPipeline.h"
#include "../include/Application.h"
#include "../include/Mesh.h"
#include "../include/MeshLoader.h"
#include "../include/CoreTime.h"

CRenderer::CRenderer(GGameObject & GameObject) : CComponent(L"Renderer", GameObject), Model(NULL) {
}

bool CRenderer::Initialize() {
	RenderStage * TestStage = Application::GetInstance().GetRenderPipeline().GetStage(L"TestStage");
	if (TestStage != NULL) {
		TestStage->OnRenderEvent.AttachObserver(&TestStageObserver);
		TestStageObserver.AddCallback("Render", std::bind(&CRenderer::Render, this));
	}

	Debug::Log(Debug::LogDebug, L"Renderer '%ls'[%d] Initalized", GetIdentifierName().c_str(), GetIdentifierHash());

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
	Debug::Log(Debug::LogDebug, L"Renderer '%ls'[%d] Destroyed", GetIdentifierName().c_str(), GetIdentifierHash());
}

void CRenderer::Render() {
	if (Model == NULL) return;

	RenderStage * Stage = Application::GetInstance().GetRenderPipeline().GetStage(L"TestStage");
	if (Stage == NULL) return;

	if (Model->SetUpBuffers()) {
		Model->BindSubdivisionVertexArray((int)fmodf(Time::GetEpochTimeSeconds(), (float)Model->Data.Materials.size()));

		Material * CurrentMaterial = Stage->CurrentMaterial;
		CurrentMaterial->SetAttribMatrix4x4Array("_iModelMatrix", 1, &GetGameObject().Transformation.GetLocalToWorldMatrix(), Stage->GetMatrixBuffer());
		Model->DrawSubdivisionInstanciated(1, (int)fmodf(Time::GetEpochTimeSeconds(), (float)Model->Data.Materials.size()));
	}
}
