#pragma once

namespace EmptySource {

	class RenderingAPI {
	public:
		enum class API {
			None = 0, OpenGL = 1, Vulkan = 2
		};

		virtual void ClearCurrentRender(bool bClearColor, const Vector4& Color, bool bClearDepth, float Depth, bool bClearStencil, unsigned int Stencil) = 0;

		virtual void SetViewport(const Box2D& Viewport) = 0;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray, unsigned int Count = 1) = 0;

		inline static API GetAPI() {
			return AppInterface;
		};

	private:
		static API AppInterface;

	};

}