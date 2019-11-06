
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/Window.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/RenderStage.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

#include "Resources/ShaderManager.h"
#include "Resources/TextureManager.h"
#include "Rendering/MeshPrimitives.h"

#include "Utility/TextFormattingMath.h"

#include "..\Public\RenderStageFirst.h"

using namespace ESource;

RenderStageFirst::RenderStageFirst(const IName & Name, RenderPipeline * Pipeline) : RenderStage(Name, Pipeline) {
}

void RenderStageFirst::End() {
	if (Target == NULL) return;
	Target->Bind();
	Target->Clear();
	Target->Unbind();

	Rendering::SetAlphaBlending(BF_None, BF_None);
	Scene.RenderLightMap(0, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
	Scene.RenderLightMap(1, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
	Rendering::SetAlphaBlending(BF_None, BF_None);
	Rendering::SetViewport(GeometryBufferTarget->GetViewport());
	GeometryBufferTarget->Bind();
	GeometryBufferTarget->Clear();
	Scene.DeferredRenderOpaque();
	Rendering::Flush();
	GeometryBufferTarget->TransferBitsTo(
		&*Target, true, true, true, FM_MinMagNearest,
		Target->GetViewport(), GeometryBufferTarget->GetViewport()
	);
	GeometryBufferTarget->Bind();
	Scene.DeferredRenderTransparent();
	GeometryBufferTarget->Unbind();
	Rendering::Flush();

	Rendering::SetAlphaBlending(BF_SrcAlpha, BF_OneMinusSrcAlpha);
	Rendering::SetViewport(Target->GetViewport());
	Target->Bind();
	Scene.ForwardRender();
	Rendering::Flush();
	Target->Unbind();
}

void RenderStageFirst::Begin() {
	Scene.Clear();
}
