#pragma once

struct FileStream;

enum ShaderType {
	Vertex, Fragment, Pixel, Geometry
};

class Shader {
private:
	FileStream* VertexStream;
	FileStream* FragmentStream;
	unsigned int VertexShader;
	unsigned int FragmentShader;
	unsigned int ShaderProgram;
	WString Name;

	bool bIsLinked;

	//* Create and compile our GLSL shader program from text files
	bool Compile(ShaderType Type);

	//* Link the shader to OpenGL
	bool LinkProgram();

public:

	Shader();

	//* Create shader with name
	Shader(const WString & Name);

	//* Add shader to shader program
	void LoadShader(ShaderType Type, WString ShaderPath);

	//* Compile shader program
	void Compile();

	//* Get the location id of a uniform variable in this shader
	unsigned int GetLocationID(const Char* LocationName) const;

	//* Unloads the shader program
	void Unload();

	//* Prepare OpenGL to use this shader
	void Use() const;

	//* The shader is valid for use?
	bool IsValid() const;
};