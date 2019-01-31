#pragma once

struct FileStream;

class Shader {
public:
	enum Type {
		Vertex = GL_VERTEX_SHADER,
		Fragment = GL_FRAGMENT_SHADER,
		Compute = GL_COMPUTE_SHADER,
		Geometry = GL_GEOMETRY_SHADER
	};

private:
	Type ShaderType;
	FileStream* ShaderCode;
	unsigned int ShaderUnit;

	//* Create and compile our GLSL shader unit
	bool Compile();

public:

	Shader();

	//* Create shader with name
	Shader(Type ShaderType, FileStream* ShaderPath);

	// Get the shader unit identifier
	unsigned int GetShaderUnit() const;

	// Get the shader type
	Type GetType() const;

	//* Unloads the shader unit
	void Unload();

	//* The shader is valid for use?
	bool IsValid() const;
};