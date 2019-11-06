
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

#include "../Public/RenderStageSecond.h"

using namespace ESource;

RenderStageSecond::RenderStageSecond(const IName & Name, RenderPipeline * Pipeline) : RenderStage(Name, Pipeline) {
	auto & TextureMng = TextureManager::GetInstance();
	MainScreenColorTexture = TextureMng.CreateTexture2D(L"MainScreenColor2", L"", PF_RGB16F, FM_MinMagNearest, SAM_Clamp, Pipeline->GetRenderSize());
	MainScreenColorTexture->Load();
	GeometryBufferTextures[GB_Depth] = TextureMng.CreateTexture2D(L"GBDepth2", L"", PF_DepthComponent24, FM_MinMagNearest, SAM_Clamp, Pipeline->GetRenderSize());
	GeometryBufferTextures[GB_Normal] = TextureMng.CreateTexture2D(L"GBNormal2", L"", PF_RG16F, FM_MinMagNearest, SAM_Clamp, Pipeline->GetRenderSize());
	GeometryBufferTextures[GB_Depth]->Load();
	GeometryBufferTextures[GB_Normal]->Load();

	MainScreenTarget = RenderTarget::Create();
	MainScreenTarget->BindTexture2D((Texture2D *)MainScreenColorTexture->GetTexture(), Pipeline->GetRenderSize());
	MainScreenTarget->CreateRenderDepthBuffer2D(PF_DepthComponent24, Pipeline->GetRenderSize());

	GeometryBufferTarget2 = RenderTarget::Create();
	Texture2D * Buffers[1] = {
		(Texture2D *)GeometryBufferTextures[GB_Normal]->GetTexture()
	};
	int Lods[1] = { 0 };
	int Attachments[1] = { 0 };
	GeometryBufferTarget2->BindDepthTexture2D((Texture2D *)GeometryBufferTextures[GB_Depth]->GetTexture(), Pipeline->GetRenderSize(), 0);
	GeometryBufferTarget2->BindTextures2D(Buffers, Pipeline->GetRenderSize(), Lods, Attachments, 1);
}

void RenderStageSecond::End() {
	if (Target == NULL) return;
	MainScreenTarget->Bind();
	MainScreenTarget->Clear();
	MainScreenTarget->Unbind();

	Rendering::SetAlphaBlending(BF_None, BF_None);
	Scene.RenderLightMap(0, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
	Scene.RenderLightMap(1, ShaderManager::GetInstance().GetProgram(L"DepthTestShader"));
	Rendering::SetAlphaBlending(BF_None, BF_None);
	Rendering::SetViewport(GeometryBufferTarget2->GetViewport());
	GeometryBufferTarget2->Bind();
	GeometryBufferTarget2->Clear();
	Scene.DeferredRenderOpaque();
	Rendering::Flush();
	GeometryBufferTarget2->TransferBitsTo(
		&*MainScreenTarget, true, true, true, FM_MinMagNearest,
		MainScreenTarget->GetViewport(), GeometryBufferTarget2->GetViewport()
	);
	GeometryBufferTarget2->Bind();
	Scene.DeferredRenderTransparent();
	GeometryBufferTarget2->Unbind();
	Rendering::Flush();

	Rendering::SetAlphaBlending(BF_SrcAlpha, BF_OneMinusSrcAlpha);
	Rendering::SetViewport(MainScreenTarget->GetViewport());
	MainScreenTarget->Bind();
	Scene.ForwardRender();
	Rendering::Flush();
	MainScreenTarget->Unbind();
	Rendering::SetAlphaBlending(BF_None, BF_None);

	RShaderPtr BloomThresholdShader = ShaderManager::GetInstance().GetProgram(L"PostProcessingBloomThreshold");
	RShaderPtr BloomShader = ShaderManager::GetInstance().GetProgram(L"PostProcessingBloom");
	RShaderPtr RenderShader = ShaderManager::GetInstance().GetProgram(L"RenderScreenTexture");
	RTexturePtr BloomThresholdTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloomThreshold", L"", PF_RGB16F, FM_MinMagLinear, SAM_Clamp, 0);
	if (BloomThresholdTexture) {
		if (BloomThresholdTexture->GetSize() != Target->GetSize() / 2) {
			BloomThresholdTexture->Unload();
			BloomThresholdTexture->SetSize(Target->GetSize() / 2);
			BloomThresholdTexture->Load();
		}
	}

	Rendering::SetViewport({ 0, 0, BloomThresholdTexture->GetSize().X, BloomThresholdTexture->GetSize().Y });

	static RenderTargetPtr BloomThresholdTarget = RenderTarget::Create();
	BloomThresholdTarget->BindTexture2D((Texture2D *)BloomThresholdTexture->GetTexture(), BloomThresholdTexture->GetSize());
	BloomThresholdTarget->Bind();
	BloomThresholdTarget->Clear();
	if (BloomThresholdShader && BloomThresholdShader->IsValid()) {
		BloomThresholdShader->GetProgram()->Bind();
		Rendering::SetActiveDepthTest(false);
		Rendering::SetDepthFunction(DF_Always);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetCullMode(CM_None);
		BloomThresholdShader->GetProgram()->SetTexture("_MainTexture", MainScreenTarget->GetBindedTexture(0), 0);
		const float Treshold = 1.0F;
		BloomThresholdShader->GetProgram()->SetFloat1Array("_Threshold", &Treshold);
		MeshPrimitives::Quad.GetVertexArray()->Bind();
		Rendering::DrawIndexed(MeshPrimitives::Quad.GetVertexArray());
		BloomThresholdShader->GetProgram()->Unbind();
	}
	BloomThresholdTarget->Unbind();
	Rendering::Flush();
	BloomThresholdTexture->DeleteMipMaps();
	BloomThresholdTexture->GenerateMipMaps();

	RTexturePtr BloomHorizontalTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloomHorizontalPass", L"", PF_RGB16F, FM_MinMagLinear, SAM_Clamp, 0);
	if (BloomHorizontalTexture) {
		if (BloomHorizontalTexture->GetSize() != BloomThresholdTexture->GetSize()) {
			BloomHorizontalTexture->Unload();
			BloomHorizontalTexture->SetSize(BloomThresholdTexture->GetSize());
			BloomHorizontalTexture->Load();
		}
	}
	RTexturePtr BloomTexture = TextureManager::GetInstance().CreateTexture2D(L"PPBloom", L"", PF_RGB16F, FM_MinMagNearest, SAM_Clamp, 0);
	if (BloomTexture) {
		if (BloomTexture->GetSize() != BloomThresholdTexture->GetSize()) {
			BloomTexture->Unload();
			BloomTexture->SetSize(BloomThresholdTexture->GetSize());
			BloomTexture->Load();
		}
	}

	if (BloomShader && BloomShader->IsValid()) {
		BloomShader->GetProgram()->Bind();

		Vector2 Radius(10.0F, 10.0F);
		Vector2 HorizontalDirection(1.0F, 0.0F);
		Vector2 VerticalDirection(0.0F, 1.0F);

		static RenderTargetPtr BloomHorizontalBlurTarget = RenderTarget::Create();
		BloomHorizontalBlurTarget->BindTexture2D((Texture2D *)BloomHorizontalTexture->GetTexture(), BloomHorizontalTexture->GetSize());
		BloomHorizontalBlurTarget->Bind();
		BloomHorizontalBlurTarget->Clear();
		Rendering::SetActiveDepthTest(false);
		Rendering::SetDepthFunction(DF_Always);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetCullMode(CM_None);
		BloomShader->GetProgram()->SetTexture("_MainTexture", BloomThresholdTexture->GetTexture(), 0);
		BloomShader->GetProgram()->SetFloat1Array("_Radius", &Radius[0], 1);
		BloomShader->GetProgram()->SetFloat2Array("_Direction", HorizontalDirection.PointerToValue(), 1);
		BloomShader->GetProgram()->SetFloat2Array("_Resolution", Vector2(BloomThresholdTexture->GetSize()).PointerToValue(), 1);
		MeshPrimitives::Quad.GetVertexArray()->Bind();
		Rendering::DrawIndexed(MeshPrimitives::Quad.GetVertexArray());
		BloomHorizontalBlurTarget->Unbind();
		Rendering::Flush();
		BloomHorizontalTexture->DeleteMipMaps();
		BloomHorizontalTexture->GenerateMipMaps();

		static RenderTargetPtr BloomBlurTarget = RenderTarget::Create();
		BloomBlurTarget->BindTexture2D((Texture2D *)BloomTexture->GetTexture(), BloomTexture->GetSize());
		BloomBlurTarget->Bind();
		BloomBlurTarget->Clear();
		Rendering::SetActiveDepthTest(false);
		Rendering::SetDepthFunction(DF_Always);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetCullMode(CM_None);
		BloomShader->GetProgram()->SetTexture("_MainTexture", BloomHorizontalTexture->GetTexture(), 0);
		BloomShader->GetProgram()->SetFloat1Array("_Radius", &Radius[1], 1);
		BloomShader->GetProgram()->SetFloat2Array("_Direction", VerticalDirection.PointerToValue(), 1);
		BloomShader->GetProgram()->SetFloat2Array("_Resolution", Vector2(BloomHorizontalTexture->GetSize()).PointerToValue(), 1);

		MeshPrimitives::Quad.GetVertexArray()->Bind();
		Rendering::DrawIndexed(MeshPrimitives::Quad.GetVertexArray());
		BloomBlurTarget->Unbind();
		BloomShader->GetProgram()->Unbind();
		Rendering::Flush();
	}
	Rendering::Flush();

	RShaderPtr SSAOShader = ShaderManager::GetInstance().GetProgram(L"PostProcessingSSAO");
	RShaderPtr SSAOShaderBlur = ShaderManager::GetInstance().GetProgram(L"PostProcessingSSAOBlur");

	RTexturePtr SSAOTexture = TextureManager::GetInstance().CreateTexture2D(L"PPSSAO", L"", PF_R32F, FM_MinMagNearest, SAM_Repeat);
	if (SSAOTexture) {
		if (SSAOTexture->GetSize() != Target->GetSize() / IntVector3(2, 2, 1)) {
			SSAOTexture->Unload();
			SSAOTexture->SetSize(Target->GetSize() / IntVector3(2, 2, 1));
			SSAOTexture->Load();
		}
	}

	RTexturePtr SSAOBlurTexture = TextureManager::GetInstance().CreateTexture2D(L"PPSSAOBlur", L"", PF_R8, FM_MinMagNearest, SAM_Repeat);
	if (SSAOBlurTexture) {
		if (SSAOBlurTexture->GetSize() != Target->GetSize() / IntVector3(2, 2, 1)) {
			SSAOBlurTexture->Unload();
			PixelMap WhiteData(Target->GetSize().X / 2, Target->GetSize().Y / 2, 1, PF_R8);
			PixelMapUtility::PerPixelOperator(WhiteData, [](unsigned char * Pixel, const unsigned char &) { Pixel[0] = 255; });
			SSAOBlurTexture->SetPixelData(WhiteData);
			SSAOBlurTexture->Load();
		}
	}

	Rendering::SetViewport({ 0, 0, (int)SSAOTexture->GetSize().X, (int)SSAOTexture->GetSize().Y });

	static RenderTargetPtr SSAOTarget = RenderTarget::Create();
	SSAOTarget->BindTexture2D((Texture2D *)SSAOTexture->GetTexture(), SSAOTexture->GetSize());
	SSAOTarget->Bind();
	SSAOTarget->Clear();
	if (SSAOShader && SSAOShader->IsValid()) {
		SSAOShader->GetProgram()->Bind();
		Rendering::SetActiveDepthTest(false);
		Rendering::SetDepthFunction(DF_Always);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetCullMode(CM_None);
		SSAOShader->GetProgram()->SetFloat3Array("_Kernel", (float *)&Application::GetInstance()->GetRenderPipeline().SSAOKernel[0], 64);
		SSAOShader->GetProgram()->SetMatrix4x4Array("_ProjectionMatrix", Scene.ProjectionMatrix.PointerToValue());
		SSAOShader->GetProgram()->SetMatrix4x4Array("_ViewMatrix", Scene.EyeTransform.GetGLViewMatrix().PointerToValue());
		SSAOShader->GetProgram()->SetFloat2Array("_NoiseScale", (SSAOTexture->GetSize().FloatVector3() / 4.0F).PointerToValue());
		SSAOShader->GetProgram()->SetTexture("_GDepth", GeometryBufferTextures[GB_Depth]->GetTexture(), 0);
		SSAOShader->GetProgram()->SetTexture("_GNormal", GeometryBufferTextures[GB_Normal]->GetTexture(), 1);
		SSAOShader->GetProgram()->SetTexture("_NoiseTexture", TextureManager::GetInstance().GetTexture(L"SSAONoise")->GetTexture(), 2);
		MeshPrimitives::Quad.GetVertexArray()->Bind();
		Rendering::DrawIndexed(MeshPrimitives::Quad.GetVertexArray());
	}
	SSAOTarget->Unbind();
	Rendering::Flush();

	static RenderTargetPtr SSAOBlurTarget = RenderTarget::Create();
	SSAOBlurTarget->BindTexture2D((Texture2D *)SSAOBlurTexture->GetTexture(), SSAOBlurTexture->GetSize());
	SSAOBlurTarget->Bind();
	SSAOBlurTarget->Clear();
	if (SSAOShaderBlur && SSAOShaderBlur->IsValid()) {
		SSAOShaderBlur->GetProgram()->Bind();
		Rendering::SetActiveDepthTest(false);
		Rendering::SetDepthFunction(DF_Always);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetCullMode(CM_None);
		SSAOShaderBlur->GetProgram()->SetTexture("_SSAO", SSAOTexture->GetTexture(), 0);
		MeshPrimitives::Quad.GetVertexArray()->Bind();
		Rendering::DrawIndexed(MeshPrimitives::Quad.GetVertexArray());
	}
	SSAOBlurTarget->Unbind();
	Rendering::Flush();

	MainScreenTarget->TransferBitsTo(
		NULL, false, true, true, FM_MinMagNearest,
		MainScreenTarget->GetViewport(), Application::GetInstance()->GetWindow().GetViewport()
	);
	Rendering::SetDefaultRender();
	Rendering::SetAlphaBlending(BF_SrcAlpha, BF_OneMinusSrcAlpha);
	Rendering::SetViewport(Application::GetInstance()->GetWindow().GetViewport());

	if (RenderShader && RenderShader->IsValid()) {
		RenderShader->GetProgram()->Bind();
		Rendering::SetActiveDepthTest(false);
		Rendering::SetDepthFunction(DF_Always);
		Rendering::SetRasterizerFillMode(FM_Solid);
		Rendering::SetCullMode(CM_None);

		RenderShader->GetProgram()->SetTexture("_MainTexture", MainScreenTarget->GetBindedTexture(0), 0);
		RenderShader->GetProgram()->SetTexture("_BloomTexture", BloomTexture->GetTexture(), 1);
		RenderShader->GetProgram()->SetTexture("_AOTexture", SSAOBlurTexture->GetTexture(), 2);
		RenderShader->GetProgram()->SetTexture("_DepthTexture", GeometryBufferTextures[GB_Depth]->GetTexture(), 3);
		RenderShader->GetProgram()->SetFloat1Array("_Exposure", &Pipeline->Exposure, 1);
		RenderShader->GetProgram()->SetFloat1Array("_Gamma", &Pipeline->Gamma, 1);

		MeshPrimitives::Quad.GetVertexArray()->Bind();
		Rendering::DrawIndexed(MeshPrimitives::Quad.GetVertexArray());
		RenderShader->GetProgram()->Unbind();
	}
}

void RenderStageSecond::Begin() {
	Scene.Clear(); 
	
	if (MainScreenTarget->GetSize() != IntVector3(Pipeline->GetRenderSize(), 1)) {
		MainScreenTarget.reset();
		MainScreenColorTexture->Unload();
		MainScreenColorTexture->SetSize(Pipeline->GetRenderSize());
		MainScreenColorTexture->Load();
		for (char i = 0; i < 2; ++i) {
			GeometryBufferTextures[i]->Unload();
			GeometryBufferTextures[i]->SetSize(Pipeline->GetRenderSize());
			GeometryBufferTextures[i]->Load();
		}

		MainScreenTarget = RenderTarget::Create();
		MainScreenTarget->BindTexture2D((Texture2D *)MainScreenColorTexture->GetTexture(), Pipeline->GetRenderSize());
		MainScreenTarget->CreateRenderDepthBuffer2D(PF_DepthComponent24, Pipeline->GetRenderSize());
		MainScreenTarget->Unbind();

		GeometryBufferTarget2 = RenderTarget::Create();
		Texture2D * Buffers[1] = {
			(Texture2D *)GeometryBufferTextures[GB_Normal]->GetTexture()
		};
		int Lods[1] = { 0 };
		int Attachments[1] = { 0 };
		GeometryBufferTarget2->BindTextures2D(Buffers, Pipeline->GetRenderSize(), Lods, Attachments, 1);
		GeometryBufferTarget2->BindDepthTexture2D((Texture2D *)GeometryBufferTextures[GB_Depth]->GetTexture(), Pipeline->GetRenderSize(), 0);
	}
}
