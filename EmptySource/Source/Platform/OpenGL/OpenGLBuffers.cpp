
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/RenderingAPI.h"
#include "Rendering/Mesh.h"

#include "Platform/OpenGL/OpenGLBuffers.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include "glad/glad.h"

namespace EmptySource {

	// Vertex Buffer //

	OpenGLVertexBuffer::OpenGLVertexBuffer(float * Vertices, unsigned int Size, EUsageMode Usage)
		: Size(Size), Usage(Usage), Layout()
	{
		glCreateBuffers(1, &VertexBufferID);
		glBindBuffer(GL_ARRAY_BUFFER, VertexBufferID);
		glBufferData(GL_ARRAY_BUFFER, Size, Vertices, OpenGLAPI::UsageModeToDrawBaseType(Usage));
	}

	OpenGLVertexBuffer::~OpenGLVertexBuffer() {
		Unbind();
		glDeleteBuffers(1, &VertexBufferID);
	}

	void OpenGLVertexBuffer::Bind() const {
		glBindBuffer(GL_ARRAY_BUFFER, VertexBufferID);
	}

	void OpenGLVertexBuffer::Unbind() const {
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void * OpenGLVertexBuffer::GetNativeObject() {
		return (void *)(unsigned long long)VertexBufferID;
	}

	// Index Buffer //

	OpenGLIndexBuffer::OpenGLIndexBuffer(unsigned int * Indices, unsigned int Count, EUsageMode Usage)
		: Size(Count * sizeof(unsigned int)), Usage(Usage)
	{
		glCreateBuffers(1, &IndexBufferID);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, Size, Indices, OpenGLAPI::UsageModeToDrawBaseType(Usage));
	}

	OpenGLIndexBuffer::~OpenGLIndexBuffer() {
		Unbind();
		glDeleteBuffers(1, &IndexBufferID);
	}

	void OpenGLIndexBuffer::Bind() const {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferID);
	}

	void OpenGLIndexBuffer::Unbind() const {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	void * OpenGLIndexBuffer::GetNativeObject() {
		return (void *)(unsigned long long)IndexBufferID;
	}

	// Vertex Array //

	OpenGLVertexArray::OpenGLVertexArray() {
		glCreateVertexArrays(1, &VertexArrayID);
	}

	OpenGLVertexArray::~OpenGLVertexArray() {
		Unbind();
		glDeleteVertexArrays(1, &VertexArrayID);
	}

	void OpenGLVertexArray::Bind() const {
		glBindVertexArray(VertexArrayID);
	}

	void OpenGLVertexArray::Unbind() const {
		glBindVertexArray(0);
	}

	void * OpenGLVertexArray::GetNativeObject() {
		return (void *)(unsigned long long)VertexArrayID;
	}

	void OpenGLVertexArray::AddVertexBuffer(const VertexBufferPtr & Buffer) {
		ES_CORE_ASSERT(Buffer->GetLayout().GetElements().size(), "Vertex Buffer has no layout!");
		glBindVertexArray(VertexArrayID);
		Buffer->Bind();

		unsigned int Index = 0;
		const auto& Layout = Buffer->GetLayout();
		for (const auto& Element : Layout) {
			glEnableVertexAttribArray(Index);
			glVertexAttribPointer(
				Index,
				Element.GetElementCount(),
				OpenGLAPI::ShaderDataTypeToBaseType(Element.DataType),
				Element.Normalized ? GL_TRUE : GL_FALSE,
				Layout.GetStride(),
				(void *)Element.Offset);
			Index++;
		}
		
		VertexBufferPointers.push_back(Buffer);
	}

	void OpenGLVertexArray::AddIndexBuffer(const IndexBufferPtr & Buffer) {
		glBindVertexArray(VertexArrayID);
		Buffer->Bind();

		IndexBufferPointer = Buffer;
	}

	IndexBufferPtr OpenGLVertexArray::GetIndexBuffer() {
		return IndexBufferPointer;
	}

}