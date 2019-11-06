#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Shader.h"
#include "Rendering/RenderingAPI.h"

namespace ESource {

	class Rendering {
	public:

		inline static RenderingAPI::API GetAPI() { return RenderingAPI::GetAPI(); };

		inline static void ClearCurrentRender(bool bClearColor, const Vector4 & Color, bool bClearDepth, float Depth, bool bClearStencil, uint32_t Stencil) {
			RendererAppInterface->ClearCurrentRender(bClearColor, Color, bClearDepth, Depth, bClearStencil, Stencil);
		}

		inline static void SetDefaultRender() {
			RendererAppInterface->SetDefaultRender();
		}

		inline static void SetViewport(const IntBox2D& Viewport) {
			RendererAppInterface->SetViewport(Viewport);
		}

		inline static void DrawIndexed(const VertexArrayPtr& VertexArray, const Subdivision & Offsets) {
			RendererAppInterface->DrawIndexed(VertexArray, Offsets);
		}

		inline static void DrawIndexed(const VertexArrayPtr& VertexArray) {
			RendererAppInterface->DrawIndexed(VertexArray);
		}

		inline static void SetAlphaBlending(EBlendFactor Source, EBlendFactor Destination) {
			RendererAppInterface->SetAlphaBlending(Source, Destination);
		}

		inline static void SetActiveDepthTest(bool Option) {
			RendererAppInterface->SetActiveDepthTest(Option);
		}

		inline static void SetDepthWritting(bool Option) {
			RendererAppInterface->SetDepthWritting(Option);
		}

		inline static void SetDepthFunction(EDepthFunction Function) {
			RendererAppInterface->SetDepthFunction(Function);
		}

		inline static void SetActiveStencilTest(bool Option) {
			RendererAppInterface->SetActiveStencilTest(Option);
		}

		inline static void SetStencilMask(uint8_t Mask) {
			RendererAppInterface->SetStencilMask(Mask);
		}

		inline static void SetStencilFunction(EStencilFunction Function, uint8_t Reference, uint8_t Mask) {
			RendererAppInterface->SetStencilFunction(Function, Reference, Mask);
		}

		inline static void SetStencilOperation(EStencilOperation Pass, EStencilOperation Fail, EStencilOperation Both) {
			RendererAppInterface->SetStencilOperation(Pass, Fail, Both);
		}

		inline static void SetCullMode(ECullMode Mode) {
			RendererAppInterface->SetCullMode(Mode);
		}

		inline static void SetRasterizerFillMode(ERasterizerFillMode Mode) {
			RendererAppInterface->SetRasterizerFillMode(Mode);
		}

		inline static void Flush() {
			RendererAppInterface->Flush();
		}

	private:

		static RenderingAPI * RendererAppInterface;
	};

}