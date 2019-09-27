#pragma once

namespace ESource {

	class RenderingAPI {
	public:
		enum class API {
			None = 0, OpenGL = 1, Vulkan = 2
		};

		virtual void ClearCurrentRender(bool bClearColor, const Vector4& Color, bool bClearDepth, float Depth, bool bClearStencil, unsigned int Stencil) = 0;

		virtual void SetDefaultRender() = 0;

		virtual void SetViewport(const Box2D& Viewport) = 0;

		virtual void SetAlphaBlending(EBlendFactor Source, EBlendFactor Destination) = 0;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray, unsigned int Count = 1) = 0;

		virtual void SetActiveDepthTest(bool Option) = 0;

		virtual void SetDepthFunction(EDepthFunction Function) = 0;

		virtual void SetCullMode(ECullMode Mode) = 0;

		virtual void SetRasterizerFillMode(ERasterizerFillMode Mode) = 0;

		inline static API GetAPI() {
			return AppInterface;
		};

	private:
		static API AppInterface;

	};

}