#pragma once

namespace EmptySource {

	unsigned int ShaderDataTypeSize(EShaderDataType Type);

	struct BufferElement {
		NString Name;
		EShaderDataType DataType;
		unsigned long long Offset;
		unsigned int Size;
		bool Normalized;

		BufferElement() {};

		BufferElement(EShaderDataType Type, const NString& Name, bool Normalized = false)
			: Name(Name), DataType(Type), Size(ShaderDataTypeSize(Type)), Offset(0), Normalized(Normalized) {
		}

		unsigned int GetElementCount() const;
	};

	class BufferLayout {
	public:

		BufferLayout() {}

		BufferLayout(const TArrayInitializer<BufferElement> BufferElements) : ElementLayouts(BufferElements) {
			CalculateElementOffsetsAndStride();
		}

		inline const TArray<BufferElement>& GetElements() const { return ElementLayouts; };

		inline unsigned int GetStride() const { return Stride; }

		TArray<BufferElement>::iterator begin() { return ElementLayouts.begin(); }
		TArray<BufferElement>::iterator end() { return ElementLayouts.end(); }
		TArray<BufferElement>::const_iterator begin() const { return ElementLayouts.begin(); }
		TArray<BufferElement>::const_iterator end() const { return ElementLayouts.end(); }

	private:
		TArray<BufferElement> ElementLayouts;
		unsigned int Stride;

		void CalculateElementOffsetsAndStride() {
			unsigned long long Offset = 0;
			Stride = 0;

			for (auto& Element : ElementLayouts) {
				Element.Offset = Offset;
				Offset += Element.Size;
				Stride += Element.Size;
			}
		}

	};

	typedef std::shared_ptr<class VertexBuffer> VertexBufferPtr;

	class VertexBuffer {
	public:
		virtual ~VertexBuffer() = default;

		//* Bind the vertex buffer
		virtual void Bind() const = 0;

		//* Unbind the index buffer
		virtual void Unbind() const = 0;

		//* The number of bytes in the vertex buffer
		virtual inline unsigned int GetSize() const = 0;

		//* The usage used for the vertex buffer
		virtual inline EUsageMode GetUsage() const = 0;

		//* The shader layout for the vertex buffer
		virtual inline const BufferLayout& GetLayout() const = 0;

		virtual void SetLayout(const BufferLayout& layout) = 0;

		static VertexBufferPtr Create(float* Vertices, unsigned int Size, EUsageMode Usage);

	};

	typedef std::shared_ptr<class IndexBuffer> IndexBufferPtr;

	class IndexBuffer {
	public:
		virtual ~IndexBuffer() = default;

		//* Bind the index buffer
		virtual void Bind() const = 0;

		//* Unbind the index buffer
		virtual void Unbind() const = 0;

		//* The number of bytes in the index buffer
		virtual inline unsigned int GetSize() const = 0;

		//* The usage used for the index buffer
		virtual inline EUsageMode GetUsage() const = 0;

		//* The number of indices in the vertex buffer
		virtual inline unsigned int GetCount() const = 0;

		static IndexBufferPtr Create(unsigned int* Indices, unsigned int Count, EUsageMode Usage);
	};

	typedef std::shared_ptr<class VertexArray> VertexArrayPtr;

	class VertexArray {
	public:

		//* Bind the vertex array
		virtual void Bind() const = 0;

		//* Unbind the index array
		virtual void Unbind() const = 0;

		//* Bind a vertex bufer to this vertex array
		virtual void AddVertexBuffer(const VertexBufferPtr & Buffer) = 0;

		//* Bind a index bufer to this vertex array
		virtual void AddIndexBuffer(const IndexBufferPtr & Buffer) = 0;

		virtual IndexBufferPtr GetIndexBuffer() = 0;

		static VertexArrayPtr Create();

	};

}