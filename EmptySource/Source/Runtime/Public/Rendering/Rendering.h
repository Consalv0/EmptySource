#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingResources.h"

namespace EmptySource {

	class Rendering {
	public:

		inline static RenderingAPI::API GetAPI() { return RenderingAPI::GetAPI(); };
		
		inline static void ClearCurrentRender(bool bClearColor, const Vector4 & Color, bool bClearDepth, float Depth, bool bClearStencil, unsigned int Stencil) {
			RendererAppInterface->ClearCurrentRender(bClearColor, Color, bClearDepth, Depth, bClearStencil, Stencil);
		}

		inline static void DrawIndexed(const VertexArrayPtr& VertexArray, unsigned int Count = 1) {
			RendererAppInterface->DrawIndexed(VertexArray, Count);
		}

	private:

		static RenderingAPI * RendererAppInterface;
	};

}