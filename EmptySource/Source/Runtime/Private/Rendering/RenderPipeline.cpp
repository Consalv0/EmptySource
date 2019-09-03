
#include "CoreMinimal.h"
#include "Core/Window.h"
#include "Core/Application.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"

// SDL 2.0.9
#include <SDL.h>
#include <SDL_opengl.h>

namespace EmptySource {

	RenderPipeline::RenderPipeline() :
		RenderSacale(1.F), RenderStages() {
	}

	RenderPipeline::~RenderPipeline() {

	}

	void RenderPipeline::Initialize() {
		Rendering::SetAlphaBlending(EBlendFactor::BF_SrcAlpha, EBlendFactor::BF_OneMinusSrcAlpha);

		for (auto & Stage : RenderStages) {
			Stage.second->Initialize();
		}
	}

	void RenderPipeline::ContextInterval(int Interval) {
		SDL_GL_SetSwapInterval(Interval);
	}

	void RenderPipeline::RunStage(WString StageName) {
		TDictionary<WString, RenderStage *>::iterator Stage;
		if ((Stage = RenderStages.find(StageName)) != RenderStages.end()) {
			Stage->second->RunStage();
		}
	}

	bool RenderPipeline::AddStage(WString StageName, RenderStage * Stage) {
		if (RenderStages.find(StageName) == RenderStages.end()) {
			RenderStages.insert(std::pair<WString, RenderStage *>(StageName, Stage));
			Stage->Pipeline = this;
			return true;
		}
		return false;
	}

	void RenderPipeline::RemoveStage(WString StageName) {
		if (RenderStages.find(StageName) != RenderStages.end()) {
			RenderStages.erase(StageName);
		}
	}

	RenderStage * RenderPipeline::GetStage(WString StageName) const {
		if (RenderStages.find(StageName) != RenderStages.end()) {
			return RenderStages.find(StageName)->second;
		}
		return NULL;
	}

	void RenderPipeline::PrepareFrame() {
		Rendering::SetDefaultRender();
		Rendering::SetViewport({ 0.F, 0.F, (float)Application::GetInstance()->GetWindow().GetWidth(), (float)Application::GetInstance()->GetWindow().GetHeight() });
		Rendering::ClearCurrentRender(true, 0.25F, true, 1, false, 0);
	}

	void RenderPipeline::EndOfFrame() {
	}

}