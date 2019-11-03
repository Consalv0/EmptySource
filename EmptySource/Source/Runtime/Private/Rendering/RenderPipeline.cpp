
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
		RenderScale(1.F), bNeedResize(true), Gamma(2.2F), Exposure(1.0F), RenderStages(), MainScreenTarget(NULL) {
	}

	RenderPipeline::~RenderPipeline() {

	}

	void RenderPipeline::Initialize() {
		Rendering::SetAlphaBlending(EBlendFactor::BF_SrcAlpha, EBlendFactor::BF_OneMinusSrcAlpha);
		auto & TextureMng = TextureManager::GetInstance();
		MainScreenColorTexture = TextureMng.CreateTexture2D(L"MainScreenColor",   L"", PF_RGB16F, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
		MainScreenColorTexture->Load();
		GeometryBufferTextures[GB_Depth    ] = TextureMng.CreateTexture2D(L"GBDepth",     L"", PF_DepthComponent24, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
		GeometryBufferTextures[GB_Normal   ] = TextureMng.CreateTexture2D(L"GBNormal",    L"", PF_RG16F, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
		GeometryBufferTextures[GB_Specular ] = TextureMng.CreateTexture2D(L"GBSpecular",  L"", PF_RGBA8, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
		GeometryBufferTextures[GB_Velocity ] = TextureMng.CreateTexture2D(L"GBVelocity",  L"", PF_RG16F, FM_MinMagNearest, SAM_Clamp, GetRenderSize());
		GeometryBufferTextures[GB_Depth    ]->Load();
		GeometryBufferTextures[GB_Normal   ]->Load();
		GeometryBufferTextures[GB_Specular ]->Load();
		GeometryBufferTextures[GB_Velocity ]->Load();

		MainScreenTarget = RenderTarget::Create();
		MainScreenTarget->BindTexture2D((Texture2D *)MainScreenColorTexture->GetTexture(), GetRenderSize());
		MainScreenTarget->CreateRenderDepthBuffer2D(PF_DepthComponent24, GetRenderSize());

		GeometryBufferTarget = RenderTarget::Create();
		Texture2D * Buffers[2] = { 
			(Texture2D *)GeometryBufferTextures[GB_Normal]->GetTexture(),
			(Texture2D *)GeometryBufferTextures[GB_Specular]->GetTexture()
		};
		int Lods[2] = { 0, 0 };
		int Attachments[2] = { 0, 1 };
		GeometryBufferTarget->BindDepthTexture2D((Texture2D *)GeometryBufferTextures[GB_Depth]->GetTexture(), GetRenderSize(), 0);
		GeometryBufferTarget->BindTextures2D(Buffers, GetRenderSize(), Lods, Attachments, 2);

		std::uniform_real_distribution<float> RandomFloats(0.0F, 1.0F);
		std::default_random_engine Generator;
		for (int i = 0; i < 64; ++i) {
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
			Stage->second->SetRenderTarget(MainScreenTarget);
			Stage->second->SetGeometryBuffer(GeometryBufferTarget);
			Stage->second->Begin();
		}
	}

	void RenderPipeline::EndStage() {
		ES_CORE_ASSERT(ActiveStage != NULL, "No RenderStage active.");
		ActiveStage->End();
		ActiveStage = NULL;
	}

	void RenderPipeline::SubmitSubmesh(const RMeshPtr & ModelPointer, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix) {
		if (ModelPointer == NULL || !ModelPointer->IsValid()) return;
		if (ModelPointer->GetVertexData().SubdivisionsMap.find(Subdivision) == ModelPointer->GetVertexData().SubdivisionsMap.end()) {
			LOG_CORE_ERROR(L"Out of bounds mesh division in Mesh: {} WithKey: {}", ModelPointer->GetName().GetDisplayName(), Subdivision); return;
		}
		ActiveStage->SubmitMesh(ModelPointer, ModelPointer->GetVertexData().SubdivisionsMap.at(Subdivision), Mat, Matrix);
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
		Size.X = int((float)Size.X * RenderScale);
		Size.Y = int((float)Size.Y * RenderScale);
		return Size;
	}

	void RenderPipeline::SetRenderScale(float Scale) {
		bNeedResize = RenderScale != Scale;
		RenderScale = Math::Clamp01(Scale);
	}

	RTexturePtr RenderPipeline::GetGBufferTexture(GBuffers Buffer) const {
		return GeometryBufferTextures[Buffer];
	}

	RTexturePtr RenderPipeline::GetMainScreenColorTexture() const {
		return MainScreenColorTexture;
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
			MainScreenTarget.reset();
			MainScreenColorTexture->Unload();
			MainScreenColorTexture->SetSize(GetRenderSize());
			MainScreenColorTexture->Load();
			for (char i = 0; i < GB_MAX; ++i) {
				GeometryBufferTextures[i]->Unload();
				GeometryBufferTextures[i]->SetSize(GetRenderSize());
				GeometryBufferTextures[i]->Load();
			}

			MainScreenTarget = RenderTarget::Create();
			MainScreenTarget->BindTexture2D((Texture2D *)MainScreenColorTexture->GetTexture(), GetRenderSize());
			MainScreenTarget->CreateRenderDepthBuffer2D(PF_DepthComponent24, GetRenderSize());
			MainScreenTarget->Unbind();

			GeometryBufferTarget = RenderTarget::Create();
			Texture2D * Buffers[2] = {
				(Texture2D *)GeometryBufferTextures[GB_Normal]->GetTexture(),
				(Texture2D *)GeometryBufferTextures[GB_Specular]->GetTexture()
			};
			int Lods[2] = { 0, 0 };
			int Attachments[2] = { 0, 1 };
			GeometryBufferTarget->BindTextures2D(Buffers, GetRenderSize(), Lods, Attachments, 2);
			GeometryBufferTarget->BindDepthTexture2D((Texture2D *)GeometryBufferTextures[GB_Depth]->GetTexture(), GetRenderSize(), 0);
			bNeedResize = false;
		}
	}

	void RenderPipeline::EndOfFrame() {
		Rendering::Flush();
		Rendering::SetDefaultRender();
		Rendering::SetViewport({ 0, 0, (int)Application::GetInstance()->GetWindow().GetWidth(), (int)Application::GetInstance()->GetWindow().GetHeight() });
	}

}