#pragma once

struct FileStream;

enum ShaderType {
	Vertex,
	Geometry,
	Fragment,
	Compute
};

class ShaderStage {
private:
	ShaderType Type;
	FileStream* ShaderCode;

	//* Shader object for a single shader stage
	unsigned int ShaderObject;

	//* Create and compile our GLSL shader unit
	bool Compile();

public:

	ShaderStage();

	//* Create shader with name
	ShaderStage(ShaderType Type, FileStream* ShaderPath);

	//* Get the shader object identifier
	unsigned int GetShaderObject() const;

	//* Get the shader type
	ShaderType GetType() const;

	//* Unloads the shader unit
	void Delete();

	//* The shader is valid for use?
	bool IsValid() const;
};