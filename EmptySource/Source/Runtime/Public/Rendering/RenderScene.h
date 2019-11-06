#pragma once

namespace ESource {
	
	class RenderScene {
	public:
		RenderScene();

		using RenderElement = std::tuple<RMeshPtr, Subdivision, Matrix4x4>;

		void Clear();

		void ForwardRender();

		void DeferredRenderOpaque();

		void DeferredRenderTransparent();

		void RenderLightMap(uint32_t LightIndex, RShaderPtr & Shader);

		void Submit(const MaterialPtr & Mat, const RMeshPtr& MeshPtr, const Subdivision & MeshSubdivision, const Matrix4x4& Matrix);

		void SubmitInstance(const MaterialPtr & Mat, const RMeshPtr& MeshPtr, const Subdivision & MeshSubdivision, const Matrix4x4& Matrix);

		VertexBufferPtr ModelMatrixBuffer;

		Transform EyeTransform;

		Matrix4x4 ProjectionMatrix;

		int LightCount;

		struct Light {
			Transform Transformation;
			Vector3 Color;
			Vector3 Direction;
			float Intensity;
			Matrix4x4 ProjectionMatrix;
			bool CastShadow;
			RTexturePtr ShadowMap;
			float ShadowBias;
		} Lights[2];
		
		TArray<MaterialPtr> SortedMaterials;
		TDictionary<MaterialPtr, TDictionary<RMeshPtr, TArray<std::tuple<Subdivision, Matrix4x4>>>> RenderElementsByMaterial;

		TDictionary<MaterialPtr, TDictionary<RMeshPtr, TArray<std::tuple<Subdivision, Matrix4x4>>>> RenderElementsInstanceByMaterial;

	};

}