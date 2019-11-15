
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
	return;
	if (Target == NULL) return;
	Target->Bind();
	Target->Clear();
	Target->Unbind();

	Rendering::SetAlphaBlending(BF_None, BF_None);
	Scene.RenderLightMap(0, MaterialManager::GetInstance().GetMaterial(L"Core/ShadowDepth"));
	Scene.RenderLightMap(1, MaterialManager::GetInstance().GetMaterial(L"Core/ShadowDepth"));
	Rendering::SetAlphaBlending(BF_None, BF_None);
	Rendering::SetViewport(GeometryBufferTarget->GetViewport());
	GeometryBufferTarget->Bind();
	GeometryBufferTarget->Clear();
	Scene.DeferredRenderOpaque(1);
	Rendering::Flush();
	GeometryBufferTarget->TransferBitsTo(
		&*Target, true, true, true, FM_MinMagNearest,
		Target->GetViewport(), GeometryBufferTarget->GetViewport()
	);
	GeometryBufferTarget->Bind();
	Scene.DeferredRenderTransparent(1);
	GeometryBufferTarget->Unbind();
	Rendering::Flush();

	Rendering::SetAlphaBlending(BF_SrcAlpha, BF_OneMinusSrcAlpha);
	Rendering::SetViewport(Target->GetViewport());
	Target->Bind();
	Scene.ForwardRender(1);
	Rendering::Flush();
	Target->Unbind();
}

void RenderStageFirst::Begin() {
	Scene.Clear();
}
