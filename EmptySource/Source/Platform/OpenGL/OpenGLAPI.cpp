
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Mesh.h"

#include "Rendering/RenderingAPI.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include <glad/glad.h>

// SDL 2.0.9
#include <SDL.h>
#include <SDL_opengl.h>

namespace ESource {

	uint32_t OpenGLAPI::UsageModeToDrawBaseType(EUsageMode Mode) {
		switch (Mode) {
			case ESource::UM_Default:    return GL_STATIC_DRAW;
			case ESource::UM_Static:     return GL_STATIC_DRAW;
			case ESource::UM_Dynamic:    return GL_DYNAMIC_DRAW;
			case ESource::UM_Inmutable:  return GL_STREAM_DRAW;
			case ESource::UM_Draw:       return GL_STREAM_DRAW;
		}

		ES_CORE_ASSERT(true, "Unknown OpenGL EUsageMode!");
		return 0;
	}
	
	uint32_t OpenGLAPI::ShaderDataTypeToBaseType(EShaderDataType Type) {
		switch (Type) {
			case EShaderDataType::Float:
			case EShaderDataType::Float2:
			case EShaderDataType::Float3:
			case EShaderDataType::Float4:
			case EShaderDataType::Mat3x3:
			case EShaderDataType::Mat4x4:
				return GL_FLOAT;
			case EShaderDataType::Int:
			case EShaderDataType::Int2:
			case EShaderDataType::Int3:
			case EShaderDataType::Int4:
				return GL_INT;
			case EShaderDataType::Bool:
				return GL_BOOL;
		}

		ES_CORE_ASSERT(true, "Unknown OpenGL EShaderDataType!");
		return 0;
	}

	uint32_t OpenGLAPI::BlendFactorToBaseType(EBlendFactor Factor) {
		switch (Factor) {
		case ESource::BF_Zero:				return GL_ZERO;
		case ESource::BF_One:				return GL_ONE;
		case ESource::BF_SrcAlpha:			return GL_SRC_ALPHA;
		case ESource::BF_SrcColor:			return GL_SRC_COLOR;
		case ESource::BF_DstAlpha:			return GL_DST_ALPHA;
		case ESource::BF_DstColor:			return GL_DST_COLOR;
		case ESource::BF_OneMinusSrcAlpha:	return GL_ONE_MINUS_SRC_ALPHA;
		case ESource::BF_OneMinusSrcColor:	return GL_ONE_MINUS_SRC_COLOR;
		case ESource::BF_OneMinusDstAlpha:	return GL_ONE_MINUS_DST_ALPHA;
		case ESource::BF_OneMinusDstColor:	return GL_ONE_MINUS_DST_COLOR;
		case ESource::BF_None:
		default:
			return GL_NONE;
		}
	}

	uint32_t OpenGLAPI::AddressModeToBaseType(ESamplerAddressMode Mode) {
		switch (Mode) {
		case SAM_Repeat: return GL_REPEAT;
		case SAM_Mirror: return GL_MIRRORED_REPEAT;
		case SAM_Clamp:  return GL_CLAMP_TO_EDGE;
		case SAM_Border: return GL_CLAMP_TO_BORDER;
		default:         return GL_NONE;
		}
	}

	uint32_t OpenGLAPI::FilterModeToBaseType(EFilterMode Mode) {
		switch (Mode) {
		case FM_MinMagLinear:
		case FM_MinLinearMagNearest:
			return GL_LINEAR;
		case FM_MinMagNearest:
		case FM_MinNearestMagLinear:
		default:
			return GL_NEAREST;
		}
	}

	uint32_t OpenGLAPI::StencilOperationToBaseType(EStencilOperation Operation) {
		switch (Operation) {
		case SO_Keep:          return GL_KEEP;
		case SO_Zero:          return GL_ZERO;
		case SO_Replace:       return GL_REPLACE;
		case SO_Increment:     return GL_INCR;
		case SO_IncrementLoop: return GL_INCR_WRAP;
		case SO_Decrement:     return GL_DECR;
		case SO_DecrementLoop: return GL_DECR_WRAP;
		case SO_Invert:        return GL_INVERT;
		default:               return GL_KEEP;
		}
	}

	void OpenGLAPI::ClearCurrentRender(bool bClearColor, const Vector4 & Color, bool bClearDepth, float Depth, bool bClearStencil, uint32_t Stencil) {
		GLbitfield ClearFlags = 0;

		if (bClearColor) {
			ClearFlags |= GL_COLOR_BUFFER_BIT;
			glClearColor(Color.R, Color.G, Color.B, Color.A);
		}
		if (bClearDepth) {
			ClearFlags |= GL_DEPTH_BUFFER_BIT;
			glClearDepth(Depth);
		}
		if (bClearStencil) {
			ClearFlags |= GL_STENCIL_BUFFER_BIT;
			glClearStencil(Stencil);
		}

		glClear(ClearFlags);
	}

	void OpenGLAPI::SetDefaultRender() {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void OpenGLAPI::SetViewport(const IntBox2D & Viewport) {
		glViewport((GLint)Viewport.GetMinPoint().X, (GLint)Viewport.GetMinPoint().Y, (GLint)Viewport.GetWidth(), (GLint)Viewport.GetHeight());
	}

	void OpenGLAPI::DrawIndexed(const VertexArrayPtr & VertexArrayPointer) {
		ES_CORE_ASSERT(VertexArrayPointer != NULL, "Can't draw VertexArrayObject, is NULL");
		ES_CORE_ASSERT(VertexArrayPointer->GetNativeObject(), "Can't draw VertexArrayObject, object is empty");
		ES_CORE_ASSERT(VertexArrayPointer->GetIndexBuffer() != NULL, "Can't draw VertexArrayObject, IndexBuffer is missing");
		glDrawElements(GL_TRIANGLES, VertexArrayPointer->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr);
	}

	void OpenGLAPI::DrawIndexed(const VertexArrayPtr & VertexArrayPointer, const Subdivision & Offsets) {
		ES_CORE_ASSERT(VertexArrayPointer != NULL, "Can't draw VertexArrayObject, is NULL");
		ES_CORE_ASSERT(VertexArrayPointer->GetNativeObject(), "Can't draw VertexArrayObject, object is empty");
		ES_CORE_ASSERT(VertexArrayPointer->GetIndexBuffer() != NULL, "Can't draw VertexArrayObject, IndexBuffer is missing");
		glDrawElementsBaseVertex(GL_TRIANGLES, Offsets.IndexCount, GL_UNSIGNED_INT, (void*)(sizeof(uint32_t) * Offsets.BaseIndex), Offsets.BaseVertex);
	}

	void OpenGLAPI::SetAlphaBlending(EBlendFactor Source, EBlendFactor Destination) {
		///////////////////////////////////////
		///////////////////////////////////////
		// This will be in the correct place //
		glPixelStorei(GL_UNPACK_ALIGNMENT, GL_TRUE);
		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
		///////////////////////////////////////

		if (BlendFactorToBaseType(Source) != GL_NONE) {
			glEnable(GL_BLEND);
			glBlendFunc(BlendFactorToBaseType(Source), BlendFactorToBaseType(Destination));
		}
		else {
			glDisable(GL_BLEND);
		}
	}

	void OpenGLAPI::SetActiveDepthTest(bool Option) {
		if (Option)
			glEnable(GL_DEPTH_TEST);
		else
			glDisable(GL_DEPTH_TEST);
	}

	void OpenGLAPI::SetDepthWritting(bool Option) {
		glDepthMask(Option ? GL_TRUE : GL_FALSE);
	}

	void OpenGLAPI::SetDepthFunction(EDepthFunction Function) {
		switch (Function) {
		case DF_Always:       glDepthFunc(GL_ALWAYS); break;
		case DF_Equal:        glDepthFunc(GL_EQUAL); break;
		case DF_Greater:      glDepthFunc(GL_GREATER); break;
		case DF_GreaterEqual: glDepthFunc(GL_GEQUAL); break;
		case DF_Less:         glDepthFunc(GL_LESS); break;
		case DF_LessEqual:    glDepthFunc(GL_LEQUAL); break;
		case DF_Never:        glDepthFunc(GL_NEVER); break;
		case DF_NotEqual:     glDepthFunc(GL_NOTEQUAL); break;
		}
	}

	void OpenGLAPI::SetActiveStencilTest(bool Option) {
		if (Option)
			glEnable(GL_STENCIL_TEST);
		else
			glDisable(GL_STENCIL_TEST);
	}

	void OpenGLAPI::SetStencilMask(uint8_t Mask) {
		glStencilMask(Mask);
	}

	void OpenGLAPI::SetStencilFunction(EStencilFunction Function, uint8_t Reference, uint8_t Mask) {
		switch (Function) {
		case SF_Always:       glStencilFunc(GL_ALWAYS, Reference, Mask); break;
		case SF_Equal:        glStencilFunc(GL_EQUAL, Reference, Mask); break;
		case SF_Greater:      glStencilFunc(GL_GREATER, Reference, Mask); break;
		case SF_GreaterEqual: glStencilFunc(GL_GEQUAL, Reference, Mask); break;
		case SF_Less:         glStencilFunc(GL_LESS, Reference, Mask); break;
		case SF_LessEqual:    glStencilFunc(GL_LEQUAL, Reference, Mask); break;
		case SF_Never:        glStencilFunc(GL_NEVER, Reference, Mask); break;
		case SF_NotEqual:     glStencilFunc(GL_NOTEQUAL, Reference, Mask); break;
		}
	}

	void OpenGLAPI::SetStencilOperation(EStencilOperation Pass, EStencilOperation Fail, EStencilOperation PassFail) {
		glStencilOp(StencilOperationToBaseType(Fail), StencilOperationToBaseType(Pass), StencilOperationToBaseType(PassFail));
	}

	void OpenGLAPI::SetCullMode(ECullMode Mode) {
		if (Mode == CM_None)
			glDisable(GL_CULL_FACE);
		else {
			glEnable(GL_CULL_FACE);
			switch (Mode) {
			case CM_ClockWise:        glCullFace(GL_FRONT); break;
			case CM_CounterClockWise: glCullFace(GL_BACK); break;
			case CM_None:
				break;
			}
		}
	}

	void OpenGLAPI::SetRasterizerFillMode(ERasterizerFillMode Mode) {
		switch (Mode) {
		case FM_Point:     glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); break;
		case FM_Wireframe: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); break;
		case FM_Solid:     glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); break;
		}
	}

	void OpenGLAPI::Flush() {
		glFlush();
	}

}