#pragma once

namespace ESource {

	class OpenGLAPI : public RenderingAPI {
	public:

		static unsigned int UsageModeToDrawBaseType(EUsageMode Mode);

		static unsigned int ShaderDataTypeToBaseType(EShaderDataType Type);

		static unsigned int BlendFactorToBaseType(EBlendFactor Factor);

		static unsigned int AddressModeToBaseType(ESamplerAddressMode Mode);

		static unsigned int FilterModeToBaseType(EFilterMode Mode);

		virtual void ClearCurrentRender(bool bClearColor, const Vector4& Color, bool bClearDepth, float Depth, bool bClearStencil, unsigned int Stencil) override;

		virtual void SetDefaultRender() override;

		virtual void SetViewport(const Box2D& Viewport) override;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray, unsigned int Offset = 0) override;

		virtual void SetAlphaBlending(EBlendFactor Source, EBlendFactor Destination) override;

		virtual void SetActiveDepthTest(bool Option) override;

		virtual void SetDepthFunction(EDepthFunction Function) override;

		virtual void SetCullMode(ECullMode Mode) override;

		virtual void SetRasterizerFillMode(ERasterizerFillMode Mode) override;

		virtual void Flush() override;

	};

}