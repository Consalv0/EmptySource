
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingResources.h"
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

		ES_CORE_ASSERT(true, L"Unknown OpenGL EUsageMode!");
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

		ES_CORE_ASSERT(true, L"Unknown OpenGL EShaderDataType!");
		return 0;
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

	void OpenGLAPI::DrawIndexed(const VertexArrayPtr & VertexArrayPointer, unsigned int Count) {
		ES_CORE_ASSERT(VertexArrayPointer == NULL, L"Can't draw VertexArrayObject, is NULL");
		ES_CORE_ASSERT(VertexArrayPointer->GetIndexBuffer() == NULL, L"Can't draw VertexArrayObject, IndexBuffer is missing");
		glDrawElementsInstanced(GL_TRIANGLES, VertexArrayPointer->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr, Count);
	}

}