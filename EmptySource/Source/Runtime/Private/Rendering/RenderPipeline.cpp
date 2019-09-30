
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
		RenderScale(1.F), RenderStages() {
	}

	RenderPipeline::~RenderPipeline() {

	}

	void RenderPipeline::Initialize() {
		Rendering::SetAlphaBlending(EBlendFactor::BF_SrcAlpha, EBlendFactor::BF_OneMinusSrcAlpha);
	}

	void RenderPipeline::ContextInterval(int Interval) {
		SDL_GL_SetSwapInterval(Interval);
	}

	void RenderPipeline::BeginStage(const IName & StageName) {
		ES_CORE_ASSERT(ActiveStage == NULL, "Render Stage '{}' already active. Use EndStage", ActiveStage->GetName().GetNarrowDisplayName().c_str());
		TDictionary<size_t, RenderStage *>::iterator Stage;
		if ((Stage = RenderStages.find(StageName.GetID())) != RenderStages.end()) {
			ActiveStage = Stage->second;
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
		Rendering::SetDefaultRender();
		Rendering::SetViewport({ 0.F, 0.F, (float)Application::GetInstance()->GetWindow().GetWidth(), (float)Application::GetInstance()->GetWindow().GetHeight() });
		Rendering::ClearCurrentRender(true, 0.25F, true, 1, false, 0);
	}

	void RenderPipeline::EndOfFrame() {
		Application::GetInstance()->GetWindow().EndFrame();
	}

}