#pragma once

#include "Events/Event.h"
#include "Core/Transform.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"
#include "Rendering/RenderScene.h"
#include "Rendering/RenderTarget.h"

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

		virtual void SubmitPointLight(const Transform & Transformation, const Vector3 & Color, const float & Intensity);

		virtual void SubmitSpotLight(const Transform & Transformation, const Vector3 & Color, const Vector3& Direction, const float & Intensity, const Matrix4x4 & Projection);

		virtual void SubmitSpotShadowMap(const RTexturePtr & Texture, const float & Bias);

		virtual void SetEyeTransform(const Transform & EyeTransform);

		virtual void SetProjectionMatrix(const Matrix4x4 & Projection);

		virtual void SetRenderTarget(const RenderTargetPtr & InTarget);

		virtual void SetGeometryBuffer(const RenderTargetPtr & InTarget);

	private:
		IName Name;

		RenderTargetPtr Target;
		RenderTargetPtr GeometryBuffer;
	};

}