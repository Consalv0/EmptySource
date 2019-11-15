#pragma once

#include "Physics/Frustrum.h"

namespace ESource {
	
	class RenderScene {
	public:
		struct Light {
			Transform Transformation;
			Vector3 Color;
			Vector3 Direction;
			float Intensity;
			Matrix4x4 ProjectionMatrix;
			bool CastShadow;
			RTexturePtr ShadowMap;
			float ShadowBias;
			Frustrum ViewFrustrum;
			uint8_t RenderMask;
		} Lights[2];

		struct Camera {
			Transform EyeTransform;
			Matrix4x4 ProjectionMatrix; 
			Frustrum ViewFrustrum;
			uint8_t RenderMask;
		} Cameras[2];

		struct RenderElement {
			Subdivision MeshSubdivision;
			Matrix4x4 Transformation;
			uint8_t RenderMask;
		};

		RenderScene();

		void Clear();

		void ForwardRender(uint8_t CameraIndex);

		void DeferredRenderOpaque(uint8_t CameraIndex);

		void DeferredRenderTransparent(uint8_t CameraIndex);

		void RenderLightMap(uint32_t LightIndex, MaterialPtr & Material);

		void SubmitPointLight(const Transform & Transformation, const Vector3 & Color, const float & Intensity, const RTexturePtr & Texture, const float & Bias, uint8_t RenderingMask);

		void SubmitSpotLight(const Transform & Transformation, const Vector3 & Color, const Vector3& Direction, const float & Intensity, const Matrix4x4 & Projection, const RTexturePtr & Texture, const float & Bias, uint8_t RenderingMask);

		void AddCamera(const Transform & EyeTransform, const Matrix4x4 & Projection, uint8_t RenderMask);

		void Submit(const MaterialPtr & Mat, const RMeshPtr& MeshPtr, const Subdivision & MeshSubdivision, const Matrix4x4& Matrix, uint8_t RenderMask);

		void SubmitInstance(const MaterialPtr & Mat, const RMeshPtr& MeshPtr, const Subdivision & MeshSubdivision, const Matrix4x4& Matrix, uint8_t RenderMask);

		VertexBufferPtr ModelMatrixBuffer;

		int LightCount;

		int CameraCount;
		
		TArray<MaterialPtr> SortedMaterials;
		TDictionary<MaterialPtr, TDictionary<RMeshPtr, TArray<RenderElement>>> RenderElementsByMeshByMaterial;
		TDictionary<MaterialPtr, TDictionary<RMeshPtr, TArray<RenderElement>>> RenderElementsInstanceByMeshByMaterial;

	};

}