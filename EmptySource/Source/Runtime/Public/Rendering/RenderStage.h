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

		uint8_t RenderingMask;

		inline const IName & GetName() const { return Name; };

	private:
		IName Name;

	protected:
		RenderTargetPtr Target;
		RenderTargetPtr GeometryBufferTarget;

		friend class RenderPipeline;

		RenderStage(const IName & Name, RenderPipeline * Pipeline);

		RenderPipeline * Pipeline;

		virtual void End();

		virtual void Begin();

		virtual void SubmitMesh(const RMeshPtr & MeshPtr, const Subdivision & MeshSubdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix, uint8_t RenderingMask);

		virtual void SubmitMeshInstance(const RMeshPtr & MeshPtr, const Subdivision & MeshSubdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix, uint8_t RenderingMask);

		virtual void SubmitPointLight(const Transform & Transformation, const Vector3 & Color, const float & Intensity, uint8_t RenderingMask);

		virtual void SubmitSpotLight(const Transform & Transformation, const Vector3 & Color, const Vector3& Direction, const float & Intensity, const Matrix4x4 & Projection, uint8_t RenderingMask);

		virtual void SubmitSpotShadowMap(const RTexturePtr & Texture, const float & Bias);

		virtual void SetCamera(const Transform & EyeTransform, const Matrix4x4 & Projection, uint8_t RenderingMask);

		virtual void SetRenderTarget(const RenderTargetPtr & InTarget);

		virtual void SetGeometryBuffer(const RenderTargetPtr & InTarget);
	};

}