#pragma once

#include "CoreTypes.h"
#include "Core/Transform.h"
#include "Resources/ModelManager.h"
#include "Rendering/Material.h"
#include "Rendering/RenderTarget.h"

namespace ESource {

	enum EGBuffers : int {
		GB_Depth     = 0,
		GB_Normal    = 1, // Using Stereographic Projection Enconding
		GB_Specular  = 2, // Contains the metalness, roughness
		GB_Velocity  = 3,
		GB_MAX       = GB_Velocity + 1
	};

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

		virtual void Begin();

		virtual void End();

		virtual void SubmitSubmesh(const RMeshPtr & MeshPointer, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix, uint8_t CullingMask);

		virtual void SubmitSubmeshInstance(const RMeshPtr & MeshPointer, int Subdivision, const MaterialPtr & Mat, const Matrix4x4 & Matrix, uint8_t CullingMask);

		virtual void SubmitSpotLight(const Transform & Position, const Vector3 & Color, const Vector3& Direction, const float & Intensity, const Matrix4x4 & Projection, const RTexturePtr & ShadowMap, const float & Bias, uint8_t CullingMask);

		virtual void SetCamera(const Transform & EyeTransform, const Matrix4x4 & Projection, uint8_t RenderingMask);

		virtual IntVector2 GetRenderSize() const;

		//* From 0.1 to 1.0
		virtual void SetRenderScale(float Scale);

		virtual RTexturePtr GetGBufferTexture(EGBuffers Buffer) const;

		virtual RTexturePtr GetMainScreenColorTexture() const;

		template <typename T>
		bool CreateStage(const IName & StageName);

		virtual void RemoveStage(const IName & StageName);

		virtual class RenderStage * GetStage(const IName & StageName) const;

		virtual void BeginFrame();

		virtual void EndOfFrame();

	protected:
		TDictionary<size_t, class RenderStage *> RenderStages;

		RenderTargetPtr MainScreenTarget;

		RenderTargetPtr GeometryBufferTarget;

		RTexturePtr MainScreenColorTexture;

		RTexturePtr GeometryBufferTextures[GB_MAX];

		// Render Scale Target
		float RenderScale;
	};

	template<typename T>
	bool RenderPipeline::CreateStage(const IName & StageName) {
		if (RenderStages.find(StageName.GetID()) == RenderStages.end()) {
			RenderStages.insert(std::pair<size_t, RenderStage *>(StageName.GetID(), new T(StageName, this)));
			return true;
		}
		return false;
	}

}