#pragma once

namespace EmptySource {

	struct FileStream;

	enum ShaderType {
		ST_Vertex,
		ST_Geometry,
		ST_Fragment,
		ST_Compute
	};

	class ShaderStage {
	private:
		ShaderType Type;

		//* Shader object for a single shader stage
		unsigned int ShaderObject;

	public:

		ShaderStage();

		//* Create shader with name
		ShaderStage(ShaderType Type);

		//* Get the shader object identifier
		unsigned int GetShaderObject() const;

		//* Get the shader type
		ShaderType GetType() const;

		//* Create and compile our shader unit
		bool CompileFromFile(const WString & FilePath);

		//* Create and compile our shader unit
		bool CompileFromText(const String & Code);

		//* Unloads the shader unit
		void Delete();

		//* The shader is valid for use?
		bool IsValid() const;
	};

}