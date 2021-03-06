#pragma once

namespace ESource {

	class RenderingAPI {
	public:
		enum class API {
			None = 0, OpenGL = 1, Vulkan = 2
		};

		virtual void ClearCurrentRender(bool bClearColor, const Vector4& Color, bool bClearDepth, float Depth, bool bClearStencil, uint32_t Stencil) = 0;

		virtual void SetDefaultRender() = 0;

		virtual void SetViewport(const IntBox2D& Viewport) = 0;

		virtual void SetAlphaBlending(EBlendFactor Source, EBlendFactor Destination) = 0;

		virtual void DrawIndexedInstanced(const VertexArrayPtr& VertexArray, const struct Subdivision & Offsets, int Count) = 0;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray) = 0;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray, const struct Subdivision & Offsets) = 0;

		virtual void SetActiveDepthTest(bool Option) = 0;

		virtual void SetDepthWritting(bool Option) = 0;

		virtual void SetDepthFunction(EDepthFunction Function) = 0;

		virtual void SetActiveStencilTest(bool Option) = 0;

		virtual void SetStencilMask(uint8_t Mask) = 0;

		virtual void SetStencilFunction(EStencilFunction Function, uint8_t Reference, uint8_t Mask) = 0;

		virtual void SetStencilOperation(EStencilOperation Pass, EStencilOperation Fail, EStencilOperation PassFail) = 0;

		virtual void SetCullMode(ECullMode Mode) = 0;

		virtual void SetRasterizerFillMode(ERasterizerFillMode Mode) = 0;

		virtual void Flush() = 0;

		inline static API GetAPI() {
			return AppInterface;
		};

	private:
		static API AppInterface;

	};

}