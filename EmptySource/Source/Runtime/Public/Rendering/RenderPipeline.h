#pragma once

#include "CoreTypes.h"

namespace EmptySource {

	class RenderPipeline {
	protected:
		TDictionary<WString, class RenderStage *> RenderStages;

	public:
		RenderPipeline();
		~RenderPipeline();

		/* Global variables */
		// Render Scale Target
		float RenderSacale;

		RenderStage * ActiveStage;

		virtual void Initialize();

		virtual void ContextInterval(int Interval);

		virtual void BeginStage(WString StageName);

		virtual void EndStage();

		virtual bool AddStage(WString StageName, RenderStage *);

		virtual void RemoveStage(WString StageName);

		virtual RenderStage * GetStage(WString StageName) const;

		virtual RenderStage * GetActiveStage() const;

		virtual void PrepareFrame();

		virtual void EndOfFrame();
	};

}