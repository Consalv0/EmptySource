#pragma once

#include "Rendering/RenderStage.h"

class RenderStageSecond : ESource::RenderStage {
public:

protected:
	friend class ESource::RenderPipeline;

	RenderStageSecond(const ESource::IName & Name, ESource::RenderPipeline * Pipeline);

	virtual void End();

	virtual void Begin();

};