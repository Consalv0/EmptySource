#pragma once

namespace ESource {
	
	class RenderScene {
	public:
		RenderScene();

		using RenderElement = std::tuple<VertexArrayPtr, Subdivision, Matrix4x4>;

		void Clear();

		void ForwardRender();

		void DeferredRenderOpaque();

		void DeferredRenderTransparent();

		void RenderLightMap(unsigned int LightIndex, RShaderPtr & Shader);

		void Submit(const MaterialPtr & Mat, const VertexArrayPtr& VertexArray, const Subdivision & MeshSubdivision, const Matrix4x4& Matrix);

		VertexBufferPtr ModelMatrixBuffer;

		Transform EyeTransform;

		Matrix4x4 ViewProjection;

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
		TDictionary<MaterialPtr, TDictionary<VertexArrayPtr, TArray<std::tuple<Subdivision, Matrix4x4>>>> RenderElementsByMaterial;

	};

}