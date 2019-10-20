#pragma once

namespace ESource {

	class OpenGLVertexBuffer : public VertexBuffer {
	public:
		OpenGLVertexBuffer(float * Vertices, uint32_t Size, EUsageMode Usage);

		virtual ~OpenGLVertexBuffer();

		//* Bind the vertex buffer
		virtual void Bind() const override;

		//* Unbind the vertex buffer
		virtual void Unbind() const override;

		virtual void * GetNativeObject() override;

		//* The number of bytes in the vertex buffer
		virtual inline uint32_t GetSize() const override { return Size; }

		//* The usage used for the vertex buffer
		virtual inline EUsageMode GetUsage() const override { return Usage; }

		//* The shader layout for the vertex buffer
		virtual inline const BufferLayout& GetLayout() const override { return Layout; };

		virtual void SetLayout(const BufferLayout& NewLayout) override { Layout = NewLayout; };

	private:
		uint32_t Size;
		EUsageMode Usage;
		BufferLayout Layout;
		uint32_t VertexBufferID;

	};

	class OpenGLIndexBuffer : public IndexBuffer {
	public:
		OpenGLIndexBuffer(uint32_t * Indices, uint32_t Count, EUsageMode Usage);

		virtual ~OpenGLIndexBuffer();

		//* Bind the index buffer
		virtual void Bind() const override;

		//* Unbind the index buffer
		virtual void Unbind() const override;

		virtual void * GetNativeObject() override;

		//* The number of bytes in the index buffer
		virtual inline uint32_t GetSize() const override { return Size; }

		//* The usage used for the index buffer
		virtual inline EUsageMode GetUsage() const override { return Usage; }

		//* The number of indices in the index buffer
		virtual inline uint32_t GetCount() const override { return Size / sizeof(uint32_t); }

	private:
		uint32_t Size;
		EUsageMode Usage;
		uint32_t IndexBufferID;

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
		uint32_t VertexArrayID;
		TArray<VertexBufferPtr> VertexBufferPointers;
		IndexBufferPtr IndexBufferPointer;
	};
}