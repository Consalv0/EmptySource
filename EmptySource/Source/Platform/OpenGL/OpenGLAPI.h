#pragma once

namespace EmptySource {

	class OpenGLAPI : public RenderingAPI {
	public:

		static unsigned int UsageModeToDrawBaseType(EUsageMode Mode);

		static unsigned int ShaderDataTypeToBaseType(EShaderDataType Type);

		virtual void ClearCurrentRender(bool bClearColor, const Vector4& Color, bool bClearDepth, float Depth, bool bClearStencil, unsigned int Stencil) override;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray, unsigned int Count = 1) override;

	};

}