#pragma once

#include "RenderingDefinitions.h"

namespace EmptySource {

	typedef std::shared_ptr<class ShaderStage> ShaderStagePtr;

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

		virtual inline WString GetName() const = 0;
		
		//* Get the shader object
		virtual void * GetShaderObject() const = 0;

		//* The shader is valid for use?
		virtual inline bool IsValid() const = 0;

		static ShaderPtr Create(const WString& Name, TArray<ShaderStagePtr> ShaderStages);

	protected:
		//* Appends shader unit to shader program
		virtual void AppendStage(ShaderStagePtr ShaderProgram) = 0;
	};

}