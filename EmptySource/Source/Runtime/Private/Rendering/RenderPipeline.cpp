
#include "CoreMinimal.h"
#include "Core/Window.h"
#include "Core/Application.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Resources/TextureManager.h"

#include <random>

// SDL 2.0.9
#include <SDL.h>
#include <SDL_opengl.h>

namespace ESource {

	RenderPipeline::RenderPipeline() :
		RenderScale(1.F), bNeedResize(true), Gamma(2.2F), Exposure(1.0F), RenderStages(), ScreenTarget(NULL) {
	}

	RenderPipeline::~RenderPipeline() {

	}

	void RenderPipeline::Initialize() {
		Rendering::SetAlphaBlending(EBlendFactor::BF_SrcAlpha, EBlendFactor::BF_OneMinusSrcAlpha);
		TextureTarget = Texture2D::Create(GetRenderSize(), PF_RGB32F, FM_MinMagNearest, SAM_Clamp);
		ScreenTarget = RenderTarget::Create();
		ScreenTarget->BindTexture2D(TextureTarget, GetRenderSize());
		ScreenTarget->CreateRenderDepthBuffer2D(PF_DepthComponent24, GetRenderSize());
		RTexturePtr GDepth    = TextureManager::GetInstance().CreateTexture2D(L"GDepth",  L"", PF_DepthComponent24, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
		GDepth->Load();
		RTexturePtr GNormal   = TextureManager::GetInstance().CreateTexture2D(L"GNormal", L"", PF_RGB16F, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
		GNormal->Load();
		RTexturePtr GAlbedo   = TextureManager::GetInstance().CreateTexture2D(L"GAlbedo", L"", PF_RGBA8,  FM_MinMagNearest, SAM_Clamp, GetRenderSize());
		GAlbedo->Load();
		GeometryBuffer = RenderTarget::Create();
		Texture2D * Buffers[2] = { (Texture2D *)GNormal->GetNativeTexture(), (Texture2D *)GAlbedo->GetNativeTexture() };
		int Lods[2] = { 0, 0 };
		int Attachments[2] = { 0, 1 };
		GeometryBuffer->BindDepthTexture2D((Texture2D *)GDepth->GetNativeTexture(), GetRenderSize(), 0);
		GeometryBuffer->BindTextures2D(Buffers, GetRenderSize(), Lods, Attachments, 2);
		// GeometryBuffer->CreateRenderDepthBuffer2D(PF_DepthComponent24, GetRenderSize());

		std::uniform_real_distribution<float> RandomFloats(0.0F, 1.0F);
		std::default_random_engine Generator;
		for (unsigned int i = 0; i < 64; ++i) {
			Vector3 Sample(RandomFloats(Generator) * 2.0F - 1.0F, RandomFloats(Generator) * 2.0F - 1.0F, RandomFloats(Generator));
			Sample.Normalize();
			Sample *= RandomFloats(Generator);
			float Scale = float(i) / 64.0F;

			// Scale samples s.t. they're more aligned to center of kernel
			Scale = Math::Mix(0.1F, 1.F, Scale * Scale);
			Sample *= Scale;
			SSAOKernel.push_back(Sample);
		}

		PixelMap SSAONoiseData(4, 4, 1, PF_RG32F);
		PixelMapUtility::PerPixelOperator(SSAONoiseData, [&RandomFloats, &Generator] (float * Pixel, const unsigned char& Channels) {
			Pixel[0] = RandomFloats(Generator) * 2.0F - 1.0F;
			Pixel[1] = RandomFloats(Generator) * 2.0F - 1.0F;
		});

		RTexturePtr SSAONoiseTexture = TextureManager::GetInstance().CreateTexture2D(L"SSAONoise", L"", PF_RG16F, FM_MinMagNearest, SAM_Repeat);
		SSAONoiseTexture->SetPixelData(SSAONoiseData);
		SSAONoiseTexture->Load();

		bNeedResize = false;
	}

	void RenderPipeline::ContextInterval(int Interval) {
		SDL_GL_SetSwapInterval(Interval);
	}

	void RenderPipeline::BeginStage(const IName & StageName) {
		ES_CORE_ASSERT(ActiveStage == NULL, "Render Stage '{}' already active. Use EndStage", ActiveStage->GetName().GetNarrowDisplayName().c_str());
		TDictionary<size_t, RenderStage *>::iterator Stage;
		if ((Stage = RenderStages.find(StageName.GetID())) != RenderStages.end()) {
			ActiveStage = Stage->second;
			Stage->second->SetRenderTarget(ScreenTarget);
			Stage->second->SetGeometryBuffer(GeometryBuffer);
			Stage->second->Begin();
		}
	}

	void RenderPipeline::EndStage() {
		ES_CORE_ASSERT(ActiveStage != NULL, "No RenderStage active.");
		ActiveStage->End();
		ActiveStage = NULL;
	}

	void RenderPipeline::SubmitMesh(const RMeshPtr & ModelPointer, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix) {
		ActiveStage->SubmitVertexArray(ModelPointer->GetSubdivisionVertexArray(Subdivision), Mat, Matrix);
	}

	void RenderPipeline::SubmitSpotLight(const Transform & Position, const Vector3 & Color, const Vector3& Direction, const float & Intensity, const Matrix4x4 & Projection) {
		ActiveStage->SubmitSpotLight(Position, Color, Direction, Intensity, Projection);
	}

	void RenderPipeline::SubmitSpotShadowMap(const RTexturePtr & ShadowMap, const float & Bias) {
		ActiveStage->SubmitSpotShadowMap(ShadowMap, Bias);
	}

	void RenderPipeline::SetEyeTransform(const Transform & EyeTransform) {
		ActiveStage->SetEyeTransform(EyeTransform);
	}

	void RenderPipeline::SetProjectionMatrix(const Matrix4x4 & Projection) {
		ActiveStage->SetProjectionMatrix(Projection);
	}

	IntVector2 RenderPipeline::GetRenderSize() const {
		IntVector2 Size = Application::GetInstance()->GetWindow().GetSize();
		Size.x = int((float)Size.x * RenderScale);
		Size.y = int((float)Size.y * RenderScale);
		return Size;
	}

	void RenderPipeline::SetRenderScale(float Scale) {
		bNeedResize = RenderScale != Scale;
		RenderScale = Math::Clamp01(Scale);
	}

	void RenderPipeline::RemoveStage(const IName & StageName) {
		if (RenderStages.find(StageName.GetID()) != RenderStages.end()) {
			RenderStages.erase(StageName.GetID());
		}
	}

	RenderStage * RenderPipeline::GetStage(const IName & StageName) const {
		if (RenderStages.find(StageName.GetID()) != RenderStages.end()) {
			return RenderStages.find(StageName.GetID())->second;
		}
		return NULL;
	}

	RenderStage * RenderPipeline::GetActiveStage() const {
		return ActiveStage;
	}

	void RenderPipeline::BeginFrame() {
		Rendering::Flush();
		Rendering::ClearCurrentRender(true, 0.15F, true, 1, false, 0);
		if (bNeedResize) {
			ScreenTarget.reset();
			delete TextureTarget;
			TextureTarget = Texture2D::Create(GetRenderSize(), PF_RGB32F, FM_MinMagNearest, SAM_Clamp);
			ScreenTarget = RenderTarget::Create();
			ScreenTarget->BindTexture2D(TextureTarget, GetRenderSize());
			ScreenTarget->CreateRenderDepthBuffer2D(PF_DepthComponent24, GetRenderSize());
			ScreenTarget->Unbind();
			RTexturePtr GDepth = TextureManager::GetInstance().CreateTexture2D(L"GDepth", L"", PF_DepthComponent24, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
			GDepth->Unload();
			GDepth->SetSize(GetRenderSize());
			GDepth->Load();
			RTexturePtr GNormal = TextureManager::GetInstance().CreateTexture2D(L"GNormal", L"", PF_RGB16F, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
			GNormal->Unload();
			GNormal->SetSize(GetRenderSize());
			GNormal->Load();
			RTexturePtr GAlbedo = TextureManager::GetInstance().CreateTexture2D(L"GAlbedo", L"", PF_RGBA8, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
			GAlbedo->Unload();
			GAlbedo->SetSize(GetRenderSize());
			GAlbedo->Load();
			GeometryBuffer = RenderTarget::Create();
			Texture2D * Buffers[2] = { (Texture2D *)GNormal->GetNativeTexture(), (Texture2D *)GAlbedo->GetNativeTexture() };
			int Lods[2] = { 0, 0 };
			int Attachments[2] = { 0, 1 };
			GeometryBuffer->BindTextures2D(Buffers, GetRenderSize(), Lods, Attachments, 2);
			GeometryBuffer->BindDepthTexture2D((Texture2D *)GDepth->GetNativeTexture(), GetRenderSize(), 0);
			bNeedResize = false;
		}
	}

	void RenderPipeline::EndOfFrame() {
		Rendering::Flush();
		Rendering::SetDefaultRender();
		Rendering::SetViewport({ 0.F, 0.F, (float)Application::GetInstance()->GetWindow().GetWidth(), (float)Application::GetInstance()->GetWindow().GetHeight() });
	}

}