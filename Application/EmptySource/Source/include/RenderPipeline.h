#pragma once

#include "../include/CoreTypes.h"
#include "../include/RenderStage.h"

class RenderPipeline {
protected:
	TDictionary<WString, RenderStage *> RenderStages;

public:
	RenderPipeline();
	~RenderPipeline();

	/* Global variables */
	// Render Scale Target
	float RenderSacale;

	virtual void Initialize();
	
	virtual void ContextInterval(int Interval);
	
	virtual void RunStage(WString StageName);
	
	virtual bool AddStage(WString StageName, RenderStage *);

	virtual void RemoveStage(WString StageName);

	virtual RenderStage * GetStage(WString StageName) const;
	
	virtual void PrepareFrame();
	
	virtual void EndOfFrame();
};