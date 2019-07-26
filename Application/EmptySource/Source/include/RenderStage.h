#pragma once

#include "../include/Event.h"
#include "../include/Transform.h"
#include "../include/Mesh.h"
#include "../include/RenderPipeline.h"

class RenderStage {
protected:
	friend class RenderPipeline;

	RenderPipeline * Pipeline;

	unsigned int ModelMatrixBuffer;

	Transform EyeTransform;
	
	Matrix4x4 ViewProjection;

	virtual void Initialize();

	virtual void Prepare();
	
	virtual void Finish() {};

	virtual void RunStage();

public:
	// TArray<Task<>> RenderTasks;
	Event OnRenderEvent;

	class Material * CurrentMaterial;

	virtual void SetEyeTransform(const Transform & EyeTransform);

	virtual void SetViewProjection(const Matrix4x4 & Projection);

	unsigned int GetMatrixBuffer() const;
};