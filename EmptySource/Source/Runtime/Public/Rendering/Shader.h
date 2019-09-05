#pragma once

#include "RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Texture.h"

namespace EmptySource {

	typedef std::shared_ptr<class ShaderStage> ShaderStagePtr;

	struct ShaderProperty {
		NString Name;
		EShaderPropertyType Type;

		ShaderProperty(const NString Name, EShaderPropertyType Type) : Name(Name), Type(Type) {};

		ShaderProperty(const ShaderProperty& Other) {
			Type = Other.Type;
			Name = Other.Name;
		};

		ShaderProperty& operator=(const ShaderProperty & Other) {
			Type = Other.Type;
			Name = Other.Name;
			return *this;
		}
	};

	class ShaderStage {
	public:
		virtual ~ShaderStage() = default;

		//* Get the shader type
		virtual inline EShaderType GetType() const = 0;

		//* Get the shader object
		virtual void * GetStageObject() const = 0;

		//* The stage is valid
		virtual bool IsValid() const = 0;

		//* Create and compile our shader unit from a file path
		static ShaderStagePtr CreateFromFile(const WString & FilePath, EShaderType Type);

		//* Create and compile our shader unit
		static ShaderStagePtr CreateFromText(const NString & Code, EShaderType Type);
	};

	typedef std::shared_ptr<class ShaderProgram> ShaderPtr;

	class ShaderProgram {
	public:
		virtual void Bind() const = 0;

		virtual void Unbind() const = 0;

		virtual ~ShaderProgram() = default;

		//* Get the location id of a uniform variable in this shader
		virtual unsigned int GetUniformLocation(const NChar* LocationName) = 0;

		//* Get the location of the attrib in this shader
		virtual unsigned int GetAttribLocation(const NChar* LocationName) = 0;

		//* Pass Matrix4x4 Buffer Array
		virtual void SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void* Data, const VertexBufferPtr& Buffer) = 0;

		//* Pass Matrix4x4 Array
		virtual void SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass one float vector value array
		virtual void SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass one int vector value array
		virtual void SetInt1Array(const NChar * UniformName, const int * Data, const int & Count = 1) = 0;

		//* Pass two float vector value array
		virtual void SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass three float vector value array
		virtual void SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass four float vector value array
		virtual void SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass Texture 2D array
		virtual void SetTexture2D(const NChar * UniformName, TexturePtr Tex, const unsigned int& Position) = 0;

		//* Pass Cubemap array
		virtual void SetTextureCubemap(const NChar * UniformName, TexturePtr Tex, const unsigned int& Position) = 0;

		//* Get the name of the shader
		virtual inline WString GetName() const = 0;
		
		//* Get the shader object
		virtual void * GetShaderObject() const = 0;

		//* Get the chader properties
		virtual const TArray<ShaderProperty> & GetProperties() const = 0;

		//* Get the chader properties
		virtual void SetProperties(const TArrayInitializer<ShaderProperty> Properties) = 0;

		//* The shader is valid for use?
		virtual inline bool IsValid() const = 0;

		static ShaderPtr Create(const WString& Name, TArray<ShaderStagePtr> ShaderStages);

	protected:
		//* Appends shader unit to shader program
		virtual void AppendStage(ShaderStagePtr ShaderProgram) = 0;
	};

}