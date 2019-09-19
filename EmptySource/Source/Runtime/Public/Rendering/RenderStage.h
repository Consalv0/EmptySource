#pragma once

#include "Events/Event.h"
#include "Core/Transform.h"
#include "Rendering/RenderPipeline.h"
#include "Rendering/Material.h"
#include "Rendering/Mesh.h"

namespace EmptySource {

	class RenderStage {
	protected:
		friend class RenderPipeline;

		RenderPipeline * Pipeline;

		VertexBufferPtr ModelMatrixBuffer;

		Transform EyeTransform;

		Matrix4x4 ViewProjection;

		virtual void End() {};

		virtual void Begin();

	public:

		struct SceneLight {
			Point3 Position;
			Vector3 Color;
			float Intensity;
		} SceneLights[2];

		virtual void SubmitMesh(const MeshPtr & Model, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix);

		virtual void SetLight(unsigned int Index, const Point3 & Position, const Vector3 & Color, const float & Intensity);

		virtual void SetEyeTransform(const Transform & EyeTransform);

		virtual void SetProjectionMatrix(const Matrix4x4 & Projection);

		VertexBufferPtr GetMatrixBuffer() const;
	};

}