#pragma once

#include "CoreTypes.h"
#include "Core/Transform.h"
#include "Rendering/Mesh.h"
#include "Rendering/Material.h"

namespace ESource {

	class RenderPipeline {
	protected:
		TDictionary<size_t, class RenderStage *> RenderStages;

	public:
		RenderPipeline();

		~RenderPipeline();

		/* Global variables */
		// Render Scale Target
		float RenderScale;

		RenderStage * ActiveStage;

		virtual void Initialize();

		virtual void ContextInterval(int Interval);

		virtual void BeginStage(const IName & StageName);

		virtual void EndStage();

		virtual void SubmitMesh(const MeshPtr & Model, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix);

		virtual void SubmitDirectionalLight(unsigned int Index, const Transform & Position, const Vector3 & Color, const float & Intensity, const Matrix4x4 & Projection);

		virtual void SubmitDirectionalShadowMap(unsigned int Index, const RTexturePtr & ShadowMap, const float & Bias);

		virtual void SetEyeTransform(const Transform & EyeTransform);

		virtual void SetProjectionMatrix(const Matrix4x4 & Projection);

		template <typename T>
		bool CreateStage(const IName & StageName);

		virtual void RemoveStage(const IName & StageName);

		virtual RenderStage * GetStage(const IName & StageName) const;

		virtual RenderStage * GetActiveStage() const;

		virtual void BeginFrame();

		virtual void EndOfFrame();
	};

	template<typename T>
	bool RenderPipeline::CreateStage(const IName & StageName) {
		if (RenderStages.find(StageName.GetID()) == RenderStages.end()) {
			RenderStages.insert(std::pair<size_t, T *>(StageName.GetID(), new T(StageName, this)));
			return true;
		}
		return false;
	}

}