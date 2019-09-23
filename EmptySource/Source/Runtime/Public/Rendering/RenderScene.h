#pragma once

namespace EmptySource {
	
	class RenderScene {
	public:
		RenderScene();

		using RenderElement = std::tuple<VertexArrayPtr, Matrix4x4>;

		void Clear();

		void Render();

		void Submit(const MaterialPtr & Mat, const VertexArrayPtr& VertexArray, const Matrix4x4& Matrix);

		VertexBufferPtr ModelMatrixBuffer;

		Transform EyeTransform;

		Matrix4x4 ViewProjection;

		struct Light {
			Point3 Position;
			Vector3 Color;
			float Intensity;
		} Lights[2];

		TDictionary<size_t, MaterialPtr> RenderElementsMaterials;
		TDictionary<size_t, TArray<RenderElement>> RenderElementsByMaterialID;

	};

}