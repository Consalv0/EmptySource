#pragma once

namespace ESource {
	
	class RenderScene {
	public:
		RenderScene();

		using RenderElement = std::tuple<VertexArrayPtr, Matrix4x4>;

		void Clear();

		void Render();

		void RenderLightMap(unsigned int LightIndex, RShaderPtr & Shader);

		void Submit(const MaterialPtr & Mat, const VertexArrayPtr& VertexArray, const Matrix4x4& Matrix);

		VertexBufferPtr ModelMatrixBuffer;

		Transform EyeTransform;

		Matrix4x4 ViewProjection;

		struct Light {
			Transform Transformation;
			Vector3 Color;
			float Intensity;
			Matrix4x4 ProjectionMatrix;
			RTexturePtr ShadowMap;
			float ShadowBias;
		} Lights[2];

		TDictionary<size_t, MaterialPtr> RenderElementsMaterials;
		TDictionary<size_t, TArray<RenderElement>> RenderElementsByMaterialID;

	};

}