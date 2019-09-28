#pragma once

#include "Events/Event.h"
#include "Core/Transform.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"
#include "Rendering/RenderScene.h"

namespace ESource {

	class RenderStage {
	public:
		RenderScene Scene;

		inline const IName & GetName() const { return Name; };

	protected:
		friend class RenderPipeline;

		RenderStage(const IName & Name, RenderPipeline * Pipeline);

		RenderPipeline * Pipeline;

		virtual void End();

		virtual void Begin();

		virtual void SubmitVertexArray(const VertexArrayPtr & VertexArray, const MaterialPtr & Mat, const Matrix4x4 & Matrix);

		virtual void SubmitPointLight(unsigned int Index, const Transform & Transformation, const Vector3 & Color, const float & Intensity);

		virtual void SubmitDirectionalLight(unsigned int Index, const Transform & Transformation, const Vector3 & Color, const float & Intensity, const Matrix4x4 & Projection);

		virtual void SubmitDirectionalShadowMap(unsigned int Index, const RTexturePtr & Texture, const float & Bias);

		virtual void SetEyeTransform(const Transform & EyeTransform);

		virtual void SetProjectionMatrix(const Matrix4x4 & Projection);

	private:
		IName Name;
	};

}