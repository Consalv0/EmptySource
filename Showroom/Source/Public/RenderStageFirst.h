#pragma once

#include "Rendering/RenderStage.h"

class RenderStageFirst : ESource::RenderStage {
public:

protected:
	friend class ESource::RenderPipeline;

	RenderStageFirst(const ESource::IName & Name, ESource::RenderPipeline * Pipeline);

	virtual void End();

	virtual void Begin();
};