#pragma once

#include "Events/Event.h"
#include "Core/Transform.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Mesh.h"

namespace EmptySource {

	class RenderStage {
	protected:
		friend class RenderPipeline;

		RenderPipeline * Pipeline;

		VertexBufferPtr ModelMatrixBuffer;

		Transform EyeTransform;

		Matrix4x4 ViewProjection;

		virtual void Initialize();

		virtual void Prepare();

		virtual void Finish() {};

		virtual void RunStage();

	public:
		// TArray<Task<>> RenderTasks;
		// Event OnRenderEvent;

		class Material * CurrentMaterial;

		virtual void SetEyeTransform(const Transform & EyeTransform);

		virtual void SetViewProjection(const Matrix4x4 & Projection);

		VertexBufferPtr GetMatrixBuffer() const;
	};

}