
#include "../include/Application.h"
#include "../include/Window.h"
#include "../include/RenderPipeline.h"
#include "../include/GLFunctions.h"

// SDL 2.0.9
#include "../External/SDL/include/SDL.h"
#include "../External/SDL/include/SDL_opengl.h"

RenderPipeline::RenderPipeline() :
	RenderSacale(1.F), RenderStages() {
}

RenderPipeline::~RenderPipeline() {

}

void RenderPipeline::Initialize() {
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glEnable(GL_BLEND);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, CoreApplication::GetMainWindow().GetWidth(), CoreApplication::GetMainWindow().GetHeight());
	CoreApplication::GetMainWindow().ClearWindow();
}

void RenderPipeline::EndOfFrame() {
	CoreApplication::GetMainWindow().EndOfFrame();
	glBindVertexArray(0);
}
