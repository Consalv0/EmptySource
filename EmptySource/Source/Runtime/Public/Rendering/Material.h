#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	struct MaterialVariable {
		union {
			TArray<Matrix4x4> Matrix4x4Array;
			TArray<float> FloatArray;
			TArray<Vector2> Float2DArray;
			TArray<Vector3> Float3DArray;
			TArray<Vector4> Float4DArray;
			TArray<int> IntArray;
			TexturePtr Texture;
		};
		NString Name;
		EMaterialDataType VariableType;

		MaterialVariable() {};

		MaterialVariable(const MaterialVariable& Other) 
			: Matrix4x4Array(), FloatArray(), Float2DArray(), Float3DArray(), Float4DArray(), IntArray(), Texture(NULL) {
			VariableType = Other.VariableType;
			Name = Other.Name;
			switch (Other.VariableType) {
			case EmptySource::EMaterialDataType::Matrix4x4Array:
				Matrix4x4Array = Other.Matrix4x4Array; break;
			case EmptySource::EMaterialDataType::FloatArray:
				FloatArray = Other.FloatArray; break;
			case EmptySource::EMaterialDataType::Float2DArray:
				Float2DArray = Other.Float2DArray; break;
			case EmptySource::EMaterialDataType::Float3DArray:
				Float3DArray = Other.Float3DArray; break;
			case EmptySource::EMaterialDataType::Float4DArray:
				Float4DArray = Other.Float4DArray; break;
			case EmptySource::EMaterialDataType::Texture2D:
			case EmptySource::EMaterialDataType::Cubemap:
				Texture = Other.Texture; break;
			case EmptySource::EMaterialDataType::IntArray:
				IntArray = Other.IntArray; break;
			case EmptySource::EMaterialDataType::None:
			default:
				break;
			}
		};

		MaterialVariable(const NString& Name, const TArray<Matrix4x4> & Matrix4x4Array) 
			: Name(Name), VariableType(EMaterialDataType::Matrix4x4Array), Matrix4x4Array(Matrix4x4Array) { }
		MaterialVariable(const NString& Name, const TArray<float> & FloatArray) 
			: Name(Name), VariableType(EMaterialDataType::FloatArray), FloatArray(FloatArray) { }
		MaterialVariable(const NString& Name, const TArray<Vector2> & Float2DArray) 
			: Name(Name), VariableType(EMaterialDataType::Float2DArray), Float2DArray(Float2DArray) { }
		MaterialVariable(const NString& Name, const TArray<Vector3> & Float3DArray) 
			: Name(Name), VariableType(EMaterialDataType::Float3DArray), Float3DArray(Float3DArray) { }
		MaterialVariable(const NString& Name, const TArray<Vector4> & Float4DArray) 
			: Name(Name), VariableType(EMaterialDataType::Float4DArray), Float4DArray(Float4DArray) { }
		MaterialVariable(const NString& Name, const TArray<int> & IntArray) 
			: Name(Name), VariableType(EMaterialDataType::IntArray), IntArray(IntArray) { }
		MaterialVariable(const NString& Name, TexturePtr Texture, ETextureDimension Type) 
			: Name(Name), Texture(Texture) { 
			VariableType = Type == ETextureDimension::Cubemap ? EMaterialDataType::Cubemap : EMaterialDataType::Texture2D;
		}

		MaterialVariable& operator=(const MaterialVariable & Other) {
			VariableType = Other.VariableType;
			Name = Other.Name;
			switch (Other.VariableType) {
			case EmptySource::EMaterialDataType::Matrix4x4Array:
				Matrix4x4Array = Other.Matrix4x4Array; break;
			case EmptySource::EMaterialDataType::FloatArray:
				FloatArray = Other.FloatArray; break;
			case EmptySource::EMaterialDataType::Float2DArray:
				Float2DArray = Other.Float2DArray; break;
			case EmptySource::EMaterialDataType::Float3DArray:
				Float3DArray = Other.Float3DArray; break;
			case EmptySource::EMaterialDataType::Float4DArray:
				Float4DArray = Other.Float4DArray; break;
			case EmptySource::EMaterialDataType::Texture2D:
			case EmptySource::EMaterialDataType::Cubemap:
				Texture = Other.Texture; break;
			case EmptySource::EMaterialDataType::IntArray:
				IntArray = Other.IntArray; break;
			case EmptySource::EMaterialDataType::None:
			default:
				break;
			}

			return *this;
		}

		~MaterialVariable() { 
			switch (VariableType) {
			case EmptySource::EMaterialDataType::Matrix4x4Array:
				Matrix4x4Array.~vector(); break;
			case EmptySource::EMaterialDataType::FloatArray:
				FloatArray.~vector(); break;
			case EmptySource::EMaterialDataType::Float2DArray:
				Float2DArray.~vector(); break;
			case EmptySource::EMaterialDataType::Float3DArray:
				Float3DArray.~vector(); break;
			case EmptySource::EMaterialDataType::Float4DArray:
				Float4DArray.~vector(); break;
			case EmptySource::EMaterialDataType::Texture2D:
			case EmptySource::EMaterialDataType::Cubemap:
				Texture.~shared_ptr(); break;
			case EmptySource::EMaterialDataType::IntArray:
				IntArray.~vector(); break;
			case EmptySource::EMaterialDataType::None:
			default:
				break;
			}
		}
	};

	class MaterialLayout {
	public:
		MaterialLayout() {}

		MaterialLayout(const TArrayInitializer<MaterialVariable> MaterialVariables) : MaterialVariables(MaterialVariables) { }

		inline const TArray<MaterialVariable>& GetVariables() const { return MaterialVariables; };

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

		inline const MaterialLayout& GetVariables() const { return VariableLayout; };

		void SetVariables(const MaterialLayout& NewLayout) { VariableLayout = NewLayout; };

		//* Use shader program, asign uniform and render mode
		void Use() const;

	private:
		ShaderPtr MaterialShader;
		MaterialLayout VariableLayout;

		WString Name;
	};

	typedef std::shared_ptr<Material> MaterialPtr;

}