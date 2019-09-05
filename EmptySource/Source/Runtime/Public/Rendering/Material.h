#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	struct MaterialVariable : public ShaderProperty {
		union {
			TArray<Matrix4x4> Matrix4x4Array;
			TArray<float> FloatArray;
			TArray<Vector2> Float2DArray;
			TArray<Vector3> Float3DArray;
			TArray<Vector4> Float4DArray;
			TArray<int> IntArray;
			TexturePtr Texture;
		};

		MaterialVariable() : ShaderProperty("", EShaderPropertyType::None) {};

		MaterialVariable(const MaterialVariable& Other)
			: Matrix4x4Array(), FloatArray(), Float2DArray(), Float3DArray(),
			Float4DArray(), IntArray(), Texture(NULL), ShaderProperty(Other)
		{
			switch (Other.Type) {
			case EmptySource::EShaderPropertyType::Matrix4x4Array:
				Matrix4x4Array = Other.Matrix4x4Array; break;
			case EmptySource::EShaderPropertyType::FloatArray:
				FloatArray = Other.FloatArray; break;
			case EmptySource::EShaderPropertyType::Float2DArray:
				Float2DArray = Other.Float2DArray; break;
			case EmptySource::EShaderPropertyType::Float3DArray:
				Float3DArray = Other.Float3DArray; break;
			case EmptySource::EShaderPropertyType::Float4DArray:
				Float4DArray = Other.Float4DArray; break;
			case EmptySource::EShaderPropertyType::Texture2D:
			case EmptySource::EShaderPropertyType::Cubemap:
				Texture = Other.Texture; break;
			case EmptySource::EShaderPropertyType::IntArray:
				IntArray = Other.IntArray; break;
			case EmptySource::EShaderPropertyType::None:
			default:
				break;
			}
		};

		MaterialVariable(const NString& Name, const EShaderPropertyType & Type)
			: Matrix4x4Array(), FloatArray(), Float2DArray(), Float3DArray(),
			Float4DArray(), IntArray(), Texture(NULL), ShaderProperty(Name, Type) 
		{
			switch (Type) {
			case EmptySource::EShaderPropertyType::Matrix4x4Array:
				Matrix4x4Array = TArray<Matrix4x4>(1); break;
			case EmptySource::EShaderPropertyType::FloatArray:
				FloatArray = TArray<float>(1); break;
			case EmptySource::EShaderPropertyType::Float2DArray:
				Float2DArray = TArray<Vector2>(1); break;
			case EmptySource::EShaderPropertyType::Float3DArray:
				Float3DArray = TArray<Vector3>(1); break;
			case EmptySource::EShaderPropertyType::Float4DArray:
				Float4DArray = TArray<Vector4>(1); break;
			case EmptySource::EShaderPropertyType::Texture2D:
			case EmptySource::EShaderPropertyType::Cubemap:
				Texture = NULL; break;
			case EmptySource::EShaderPropertyType::IntArray:
				IntArray = TArray<int>(1); break;
			case EmptySource::EShaderPropertyType::None:
			default:
				break;
			}
		};

		MaterialVariable(const NString& Name, const TArrayInitializer<Matrix4x4> Matrix4x4Array)
			: ShaderProperty(Name, EShaderPropertyType::Matrix4x4Array), Matrix4x4Array(Matrix4x4Array) { }
		MaterialVariable(const NString& Name, const TArrayInitializer<float> FloatArray)
			: ShaderProperty(Name, EShaderPropertyType::FloatArray), FloatArray(FloatArray) { }
		MaterialVariable(const NString& Name, const TArrayInitializer<Vector2> Float2DArray)
			: ShaderProperty(Name, EShaderPropertyType::Float2DArray), Float2DArray(Float2DArray) { }
		MaterialVariable(const NString& Name, const TArrayInitializer<Vector3> Float3DArray)
			: ShaderProperty(Name, EShaderPropertyType::Float3DArray), Float3DArray(Float3DArray) { }
		MaterialVariable(const NString& Name, const TArrayInitializer<Vector4> Float4DArray)
			: ShaderProperty(Name, EShaderPropertyType::Float4DArray), Float4DArray(Float4DArray) { }
		MaterialVariable(const NString& Name, const TArrayInitializer<int> IntArray)
			: ShaderProperty(Name, EShaderPropertyType::IntArray), IntArray(IntArray) { }
		MaterialVariable(const NString& Name, TexturePtr Texture, ETextureDimension Type) 
			: ShaderProperty(Name, Type == ETextureDimension::Cubemap ? EShaderPropertyType::Cubemap : EShaderPropertyType::Texture2D),
			Texture(Texture) { }

		MaterialVariable& operator=(const MaterialVariable & Other) {
			Type = Other.Type;
			Name = Other.Name;
			switch (Other.Type) {
			case EmptySource::EShaderPropertyType::Matrix4x4Array:
				Matrix4x4Array = Other.Matrix4x4Array; break;
			case EmptySource::EShaderPropertyType::FloatArray:
				FloatArray = Other.FloatArray; break;
			case EmptySource::EShaderPropertyType::Float2DArray:
				Float2DArray = Other.Float2DArray; break;
			case EmptySource::EShaderPropertyType::Float3DArray:
				Float3DArray = Other.Float3DArray; break;
			case EmptySource::EShaderPropertyType::Float4DArray:
				Float4DArray = Other.Float4DArray; break;
			case EmptySource::EShaderPropertyType::Texture2D:
			case EmptySource::EShaderPropertyType::Cubemap:
				Texture = Other.Texture; break;
			case EmptySource::EShaderPropertyType::IntArray:
				IntArray = Other.IntArray; break;
			case EmptySource::EShaderPropertyType::None:
			default:
				break;
			}

			return *this;
		}

		~MaterialVariable() { 
			switch (Type) {
			case EmptySource::EShaderPropertyType::Matrix4x4Array:
				Matrix4x4Array.~vector(); break;
			case EmptySource::EShaderPropertyType::FloatArray:
				FloatArray.~vector(); break;
			case EmptySource::EShaderPropertyType::Float2DArray:
				Float2DArray.~vector(); break;
			case EmptySource::EShaderPropertyType::Float3DArray:
				Float3DArray.~vector(); break;
			case EmptySource::EShaderPropertyType::Float4DArray:
				Float4DArray.~vector(); break;
			case EmptySource::EShaderPropertyType::Texture2D:
			case EmptySource::EShaderPropertyType::Cubemap:
				Texture.~shared_ptr(); break;
			case EmptySource::EShaderPropertyType::IntArray:
				IntArray.~vector(); break;
			case EmptySource::EShaderPropertyType::None:
			default:
				break;
			}
		}
	};

	class MaterialLayout {
	public:
		MaterialLayout() {}

		MaterialLayout(const TArrayInitializer<MaterialVariable> MaterialVariables) : MaterialVariables(MaterialVariables) { }

		const TArray<MaterialVariable>& GetVariables() const { return MaterialVariables; };

		MaterialVariable & GetVariable(const NString & Name) const { Find(Name); };

		void SetVariable(const MaterialVariable& Variable);

		void AddVariable(const ShaderProperty& Property);

		TArray<MaterialVariable>::const_iterator Find(const NString & Name) const;

		TArray<MaterialVariable>::iterator Find(const NString & Name);

		void Clear() { MaterialVariables.clear(); };

		TArray<MaterialVariable>::iterator begin() { return MaterialVariables.begin(); }
		TArray<MaterialVariable>::iterator end() { return MaterialVariables.end(); }
		TArray<MaterialVariable>::const_iterator begin() const { return MaterialVariables.begin(); }
		TArray<MaterialVariable>::const_iterator end() const { return MaterialVariables.end(); }

	private:
		TArray<MaterialVariable> MaterialVariables;
	};

	class Material {
	public:
		bool bUseDepthTest;
		unsigned int RenderPriority;
		EDepthFunction DepthFunction;
		ERasterizerFillMode FillMode;
		ECullMode CullMode;

		Material(const WString & Name);

		//* Set material shader
		void SetShaderProgram(ShaderPtr Value);

		//* Get material shader
		ShaderPtr GetShaderProgram() const;

		WString GetName() const;

		//* Pass Matrix4x4 Buffer Array
		void SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void* Data, const VertexBufferPtr& Buffer) const;

		//* Pass Matrix4x4 Array
		void SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass one float vector value array
		void SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass one int vector value array
		void SetInt1Array(const NChar * UniformName, const int * Data, const int & Count = 1) const;

		//* Pass two float vector value array
		void SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass three float vector value array
		void SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass four float vector value array
		void SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass Texture 2D array
		void SetTexture2D(const NChar * UniformName, TexturePtr Tex, const unsigned int& Position) const;

		//* Pass Cubemap array
		void SetTextureCubemap(const NChar * UniformName, TexturePtr Tex, const unsigned int& Position) const;

		inline MaterialLayout& GetVariables() { return VariableLayout; };

		void SetVariables(const MaterialLayout& NewLayout);

		void SetProperties(const TArray<ShaderProperty>& NewLayout);

		//* Use shader program, asign uniform and render mode
		void Use() const;

	private:
		ShaderPtr MaterialShader;
		MaterialLayout VariableLayout;

		WString Name;
	};

	typedef std::shared_ptr<Material> MaterialPtr;

}