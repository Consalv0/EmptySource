#pragma once

namespace ESource {

	class OpenGLAPI : public RenderingAPI {
	public:

		static uint32_t UsageModeToDrawBaseType(EUsageMode Mode);

		static uint32_t ShaderDataTypeToBaseType(EShaderDataType Type);

		static uint32_t BlendFactorToBaseType(EBlendFactor Factor);

		static uint32_t AddressModeToBaseType(ESamplerAddressMode Mode);

		static uint32_t FilterModeToBaseType(EFilterMode Mode);

		static uint32_t StencilOperationToBaseType(EStencilOperation Operation);

		virtual void ClearCurrentRender(bool bClearColor, const Vector4& Color, bool bClearDepth, float Depth, bool bClearStencil, uint32_t Stencil) override;

		virtual void SetDefaultRender() override;

		virtual void SetViewport(const IntBox2D& Viewport) override;

		virtual void DrawIndexedInstanced(const VertexArrayPtr& VertexArray, const Subdivision & Offsets, int Count) override;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray) override;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray, const Subdivision & Offsets) override;

		virtual void SetAlphaBlending(EBlendFactor Source, EBlendFactor Destination) override;

		virtual void SetActiveDepthTest(bool Option) override;

		virtual void SetDepthWritting(bool Option) override;

		virtual void SetDepthFunction(EDepthFunction Function) override;

		virtual void SetActiveStencilTest(bool Option) override;

		virtual void SetStencilMask(uint8_t Mask) override;

		virtual void SetStencilFunction(EStencilFunction Function, uint8_t Reference, uint8_t Mask) override;

		virtual void SetStencilOperation(EStencilOperation Pass, EStencilOperation Fail, EStencilOperation PassFail) override;

		virtual void SetCullMode(ECullMode Mode) override;

		virtual void SetRasterizerFillMode(ERasterizerFillMode Mode) override;

		virtual void Flush() override;

	};

}