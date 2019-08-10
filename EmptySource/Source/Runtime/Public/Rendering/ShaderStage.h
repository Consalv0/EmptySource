#pragma once

#include "RenderingDefinitions.h"

namespace EmptySource {

	struct FileStream;

	class ShaderStage {
	private:
		EShaderType Type;

		//* Shader object for a single shader stage
		unsigned int ShaderObject;

	public:

		ShaderStage();

		//* Create shader with name
		ShaderStage(EShaderType Type);

		//* Get the shader object identifier
		unsigned int GetShaderObject() const;

		//* Get the shader type
		EShaderType GetType() const;

		//* Create and compile our shader unit
		bool CompileFromFile(const WString & FilePath);

		//* Create and compile our shader unit
		bool CompileFromText(const NString & Code);

		//* Unloads the shader unit
		void Delete();

		//* The shader is valid for use?
		bool IsValid() const;
	};

}