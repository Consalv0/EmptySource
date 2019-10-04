
#include "CoreMinimal.h"
#include "Core/Window.h"
#include "Core/Application.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"

// SDL 2.0.9
#include <SDL.h>
#include <SDL_opengl.h>

namespace ESource {

	RenderPipeline::RenderPipeline() :
		RenderScale(1.F), bNeedResize(true), Gamma(2.2), Exposure(1.0), RenderStages(), ScreenTarget(NULL) {
	}

	RenderPipeline::~RenderPipeline() {

	}

	void RenderPipeline::Initialize() {
		Rendering::SetAlphaBlending(EBlendFactor::BF_SrcAlpha, EBlendFactor::BF_OneMinusSrcAlpha);
		TextureTarget = Texture2D::Create(GetRenderSize(), PF_RGB32F, FM_MinMagNearest, SAM_Clamp);
		ScreenTarget = RenderTarget::Create();
		ScreenTarget->BindTexture2D(TextureTarget, GetRenderSize());
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
		Rendering::ClearCurrentRender(true, 0.15F, true, 1, false, 0);
		if (bNeedResize) {
			ScreenTarget.reset();
			delete TextureTarget;
			TextureTarget = Texture2D::Create(GetRenderSize(), PF_RGB32F, FM_MinMagNearest, SAM_Clamp);
			ScreenTarget = RenderTarget::Create();
			ScreenTarget->BindTexture2D(TextureTarget, GetRenderSize());
			ScreenTarget->Unbind();
			bNeedResize = false;
		}
	}

	void RenderPipeline::EndOfFrame() {
		Rendering::SetDefaultRender();
		Rendering::SetViewport({ 0.F, 0.F, (float)Application::GetInstance()->GetWindow().GetWidth(), (float)Application::GetInstance()->GetWindow().GetHeight() });
	}

}