
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"

#include "Rendering/RenderingAPI.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include <glad/glad.h>

// SDL 2.0.9
#include <SDL.h>
#include <SDL_opengl.h>

namespace EmptySource {

	unsigned int OpenGLAPI::UsageModeToDrawBaseType(EUsageMode Mode) {
		switch (Mode) {
			case EmptySource::UM_Default:    return GL_STATIC_DRAW;
			case EmptySource::UM_Static:     return GL_STATIC_DRAW;
			case EmptySource::UM_Dynamic:    return GL_DYNAMIC_DRAW;
			case EmptySource::UM_Inmutable:  return GL_STREAM_DRAW;
		}

		ES_CORE_ASSERT(true, "Unknown OpenGL EUsageMode!");
		return 0;
	}
	
	unsigned int OpenGLAPI::ShaderDataTypeToBaseType(EShaderDataType Type) {
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

	unsigned int OpenGLAPI::BlendFactorToBaseType(EBlendFactor Factor) {
		switch (Factor) {
		case EmptySource::BF_Zero:				return GL_ZERO;
		case EmptySource::BF_One:				return GL_ONE;
		case EmptySource::BF_SrcAlpha:			return GL_SRC_ALPHA;
		case EmptySource::BF_SrcColor:			return GL_SRC_COLOR;
		case EmptySource::BF_DstAlpha:			return GL_DST_ALPHA;
		case EmptySource::BF_DstColor:			return GL_DST_COLOR;
		case EmptySource::BF_OneMinusSrcAlpha:	return GL_ONE_MINUS_SRC_ALPHA;
		case EmptySource::BF_OneMinusSrcColor:	return GL_ONE_MINUS_SRC_COLOR;
		case EmptySource::BF_OneMinusDstAlpha:	return GL_ONE_MINUS_DST_ALPHA;
		case EmptySource::BF_OneMinusDstColor:	return GL_ONE_MINUS_DST_COLOR;
		case EmptySource::BF_None:
		default:
			return GL_NONE;
		}
	}

	void OpenGLAPI::ClearCurrentRender(bool bClearColor, const Vector4 & Color, bool bClearDepth, float Depth, bool bClearStencil, unsigned int Stencil) {
		GLbitfield ClearFlags = 0;

		if (bClearColor) {
			ClearFlags |= GL_COLOR_BUFFER_BIT;
			glClearColor(Color.r, Color.g, Color.b, Color.a);
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

	void OpenGLAPI::SetViewport(const Box2D & Viewport) {
		glViewport((GLint)Viewport.GetMinPoint().x, (GLint)Viewport.GetMinPoint().y, (GLint)Viewport.GetWidth(), (GLint)Viewport.GetHeight());
	}

	void OpenGLAPI::DrawIndexed(const VertexArrayPtr & VertexArrayPointer, unsigned int Count) {
		ES_CORE_ASSERT(VertexArrayPointer != NULL, "Can't draw VertexArrayObject, is NULL");
		ES_CORE_ASSERT(VertexArrayPointer->GetNativeObject(), "Can't draw VertexArrayObject, object is empty");
		ES_CORE_ASSERT(VertexArrayPointer->GetIndexBuffer() != NULL, "Can't draw VertexArrayObject, IndexBuffer is missing");
		glDrawElementsInstanced(GL_TRIANGLES, VertexArrayPointer->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr, Count);
	}

	void OpenGLAPI::SetAlphaBlending(EBlendFactor Source, EBlendFactor Destination) {
		//////////////////////////////////////
		//////////////////////////////////////
		// This will be in the correct place 
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
		///////////////////////////////////////

		if (BlendFactorToBaseType(Source)) {
			glEnable(GL_BLEND);
			glBlendFunc(BlendFactorToBaseType(Source), BlendFactorToBaseType(Destination));
		}
		else {
			glDisable(GL_BLEND);
		}
	}

	void OpenGLAPI::SetActiveDepthTest(bool Option) {
		glDepthMask(Option);
	}

	void OpenGLAPI::SetDepthFunction(EDepthFunction Function) {
		glEnable(GL_DEPTH_TEST);
		switch (Function) {
		case DF_Always:
			glDepthFunc(GL_ALWAYS); break;
		case DF_Equal:
			glDepthFunc(GL_EQUAL); break;
		case DF_Greater:
			glDepthFunc(GL_GREATER); break;
		case DF_GreaterEqual:
			glDepthFunc(GL_GEQUAL); break;
		case DF_Less:
			glDepthFunc(GL_LESS); break;
		case DF_LessEqual:
			glDepthFunc(GL_LEQUAL); break;
		case DF_Never:
			glDepthFunc(GL_NEVER); break;
		case DF_NotEqual:
			glDepthFunc(GL_NOTEQUAL); break;
		}
	}

	void OpenGLAPI::SetCullMode(ECullMode Mode) {
		if (Mode == CM_None)
			glDisable(GL_CULL_FACE);
		else {
			glEnable(GL_CULL_FACE);
			switch (Mode) {
			case CM_ClockWise:
				glCullFace(GL_FRONT); break;
			case CM_CounterClockWise:
				glCullFace(GL_BACK); break;
			case CM_None:
				break;
			}
		}
	}

	void OpenGLAPI::SetRasterizerFillMode(ERasterizerFillMode Mode) {
		switch (Mode) {
		case FM_Point:
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); break;
		case FM_Wireframe:
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); break;
		case FM_Solid:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); break;
		}
	}

}