#pragma once

namespace ESource {

	class OpenGLVertexBuffer : public VertexBuffer {
	public:
		OpenGLVertexBuffer(float * Vertices, unsigned int Size, EUsageMode Usage);

		virtual ~OpenGLVertexBuffer();

		//* Bind the vertex buffer
		virtual void Bind() const override;

		//* Unbind the vertex buffer
		virtual void Unbind() const override;

		virtual void * GetNativeObject() override;

		//* The number of bytes in the vertex buffer
		virtual inline unsigned int GetSize() const override { return Size; }

		//* The usage used for the vertex buffer
		virtual inline EUsageMode GetUsage() const override { return Usage; }

		//* The shader layout for the vertex buffer
		virtual inline const BufferLayout& GetLayout() const override { return Layout; };

		virtual void SetLayout(const BufferLayout& NewLayout) override { Layout = NewLayout; };

	private:
		unsigned int Size;
		EUsageMode Usage;
		BufferLayout Layout;
		unsigned int VertexBufferID;

	};

	class OpenGLIndexBuffer : public IndexBuffer {
	public:
		OpenGLIndexBuffer(unsigned int * Indices, unsigned int Count, EUsageMode Usage);

		virtual ~OpenGLIndexBuffer();

		//* Bind the index buffer
		virtual void Bind() const override;

		//* Unbind the index buffer
		virtual void Unbind() const override;

		virtual void * GetNativeObject() override;

		//* The number of bytes in the index buffer
		virtual inline unsigned int GetSize() const override { return Size; }

		//* The usage used for the index buffer
		virtual inline EUsageMode GetUsage() const override { return Usage; }

		//* The number of indices in the index buffer
		virtual inline unsigned int GetCount() const override { return Size / sizeof(unsigned int); }

	private:
		unsigned int Size;
		EUsageMode Usage;
		unsigned int IndexBufferID;

	};

	class OpenGLVertexArray : public VertexArray {
	public:
		OpenGLVertexArray();

		virtual ~OpenGLVertexArray();

		//* Bind the vertex buffer
		virtual void Bind() const override;

		//* Unbind the index buffer
		virtual void Unbind() const override;

		virtual void * GetNativeObject() override;

		//* Bind a vertex bufer to this vertex array
		virtual void AddVertexBuffer(const VertexBufferPtr & Buffer) override;

		//* Bind a index bufer to this vertex array
		virtual void AddIndexBuffer(const IndexBufferPtr & Buffer) override;

		virtual IndexBufferPtr GetIndexBuffer() override;

	private:
		unsigned int VertexArrayID;
		TArray<VertexBufferPtr> VertexBufferPointers;
		IndexBufferPtr IndexBufferPointer;
	};
}