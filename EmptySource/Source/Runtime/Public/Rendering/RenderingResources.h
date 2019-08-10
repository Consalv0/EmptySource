#pragma once

namespace EmptySource {

	enum class EShaderDataType {
		None = 0, Bool, Float, Float2, Float3, Float4, Int, Int2, Int3, Int4, Mat3x3, Mat4x4 
	};

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

		static VertexBuffer * Create(float* Vertices, unsigned int Size, EUsageMode Usage);

	};

	typedef std::shared_ptr<VertexBuffer> VertexBufferPtr;

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

		static IndexBuffer * Create(unsigned int* Indices, unsigned int Count, EUsageMode Usage);
	};

	typedef std::shared_ptr<IndexBuffer> IndexBufferPtr;

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

		static VertexArray * Create();

	};

	typedef std::shared_ptr<VertexArray> VertexArrayPtr;

	class RenderingAPI {
	public:

		enum class API {
			None = 0, OpenGL = 1, Vulkan = 2
		};

		virtual void ClearCurrentRender(bool bClearColor, const Vector4& Color, bool bClearDepth, float Depth, bool bClearStencil, unsigned int Stencil) = 0;

		virtual void DrawIndexed(const VertexArrayPtr& VertexArray, unsigned int Count = 1) = 0;

		inline static API GetAPI() {
			return AppInterface;
		};

	private:

		static API AppInterface;

	};

}