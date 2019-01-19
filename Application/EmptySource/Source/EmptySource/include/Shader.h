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
	FileStream* VertexStream;
	FileStream* FragmentStream;
	FileStream* ComputeStream;
	FileStream* GeometryStream;
	unsigned int VertexShader;
	unsigned int FragmentShader;
	unsigned int ComputeShader;
	unsigned int GeometryShader;
	unsigned int ShaderProgram;
	WString Name;

	bool bIsLinked;

	//* Create and compile our GLSL shader program from text files
	bool Compile(Type Type);

	//* Link the shader to OpenGL
	bool LinkProgram();

public:

	Shader();

	//* Create shader with name
	Shader(const WString & Name);

	//* Add shader to shader program
	void LoadShader(Type Type, WString ShaderPath);

	//* Compile shader program
	void Compile();

	//* Get the location id of a uniform variable in this shader
	unsigned int GetUniformLocationID(const Char* LocationName) const;

	//* Unloads the shader program
	void Unload();

	//* Prepare OpenGL to use this shader
	void Use() const;

	//* The shader is valid for use?
	bool IsValid() const;
};