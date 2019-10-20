#pragma once

#include "CoreTypes.h"
#include "Core/Transform.h"
#include "Resources/ModelManager.h"
#include "Rendering/Material.h"
#include "Rendering/RenderTarget.h"

namespace ESource {

	class RenderPipeline {
	public:
		bool bNeedResize;

		float Gamma;
		
		float Exposure;

		TArray<Vector3> SSAOKernel;

		RenderPipeline();

		~RenderPipeline();

		virtual void Initialize();

		virtual void ContextInterval(int Interval);

		virtual void BeginStage(const IName & StageName);

		virtual void EndStage();

		virtual void SubmitSubmesh(const RMeshPtr & MeshPointer, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix);

		virtual void SubmitSpotLight(const Transform & Position, const Vector3 & Color, const Vector3& Direction, const float & Intensity, const Matrix4x4 & Projection);

		virtual void SubmitSpotShadowMap(const RTexturePtr & ShadowMap, const float & Bias);

		virtual void SetEyeTransform(const Transform & EyeTransform);

		virtual void SetProjectionMatrix(const Matrix4x4 & Projection);

		virtual IntVector2 GetRenderSize() const;

		//* From 0.1 to 1.0
		virtual void SetRenderScale(float Scale);

		template <typename T>
		bool CreateStage(const IName & StageName);

		virtual void RemoveStage(const IName & StageName);

		virtual class RenderStage * GetStage(const IName & StageName) const;

		virtual class RenderStage * GetActiveStage() const;

		virtual void BeginFrame();

		virtual void EndOfFrame();

	protected:
		TDictionary<size_t, class RenderStage *> RenderStages;

		RenderTargetPtr ScreenTarget;

		RenderTargetPtr GeometryBuffer;

		Texture2D * TextureTarget;

		// Render Scale Target
		float RenderScale;

		class RenderStage * ActiveStage;
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