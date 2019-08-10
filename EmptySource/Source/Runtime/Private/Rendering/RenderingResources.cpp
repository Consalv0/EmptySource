
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingResources.h"

#include "Platform/OpenGL/OpenGLBuffers.h"

#include "Rendering/Rendering.h"

namespace EmptySource {

	RenderingAPI::API RenderingAPI::AppInterface = RenderingAPI::API::OpenGL;

	VertexBuffer * VertexBuffer::Create(float* Vertices, unsigned int Size, EUsageMode Usage) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return new OpenGLVertexBuffer(Vertices, Size, Usage);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid, can't create vertex buffer!");
			return NULL;
		}
	}

	IndexBuffer * IndexBuffer::Create(unsigned int* Indices, unsigned int Size, EUsageMode Usage) {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return new OpenGLIndexBuffer(Indices, Size, Usage);
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create index buffer!");
			return NULL;
		}
	}

	VertexArray * VertexArray::Create() {
		switch (Rendering::GetAPI()) {
		case RenderingAPI::API::OpenGL:
			return new OpenGLVertexArray();
		case RenderingAPI::API::None:
		default:
			ES_CORE_ASSERT(true, "Rendering API is not valid for this platform, can't create index buffer!");
			return NULL;
		}
	}

	unsigned int BufferElement::GetElementCount() const {
		switch (DataType) {
			case EShaderDataType::Float:
			case EShaderDataType::Int:
			case EShaderDataType::Bool:
				return 1;
			case EShaderDataType::Float2:
			case EShaderDataType::Int2:
				return 2;
			case EShaderDataType::Float3:
			case EShaderDataType::Int3:
				return 3;
			case EShaderDataType::Float4:
			case EShaderDataType::Int4:
				return 4;
			case EShaderDataType::Mat3x3:  return 3 * 3;
			case EShaderDataType::Mat4x4:  return 4 * 4;
		}

		LOG_CORE_ERROR(L"Unknown ShaderDataType: {:d}", (unsigned int)DataType);
		return 0;
	}

	unsigned int ShaderDataTypeSize(EShaderDataType Type) {
		switch (Type) {
			case EShaderDataType::Bool:     return sizeof(bool);
			case EShaderDataType::Float:    return sizeof(float);
			case EShaderDataType::Float2:   return sizeof(float) * 2;
			case EShaderDataType::Float3:   return sizeof(float) * 3;
			case EShaderDataType::Float4:   return sizeof(float) * 4;
			case EShaderDataType::Int:      return sizeof(int);
			case EShaderDataType::Int2:     return sizeof(int) * 2;
			case EShaderDataType::Int3:     return sizeof(int) * 3;
			case EShaderDataType::Int4:     return sizeof(int) * 4;
			case EShaderDataType::Mat3x3:   return sizeof(float) * 3 * 3;
			case EShaderDataType::Mat4x4:   return sizeof(float) * 4 * 4;
		}

		LOG_CORE_ERROR(L"Unknown ShaderDataType: {:d}", (unsigned int)Type);
		return 0;
	}

}
