#pragma once

class Shader;

class ShaderProgram {
private:
	Shader* VertexShader;
	Shader* FragmentShader;
	Shader* ComputeShader;
	Shader* GeometryShader;
	WString Name;

	bool bIsLinked;

	//* Link the shader to OpenGL
	bool LinkProgram();

	unsigned int Program;

public:

	ShaderProgram();

	//* Constructor with name
	ShaderProgram(WString Name);

	//* Get the location id of a uniform variable in this shader
	unsigned int GetUniformLocation(const Char* LocationName) const;

	//* Get the location of the attrib in this shader
	unsigned int GetAttribLocation(const Char* LocationName) const;

	//* Pass Matrix Array
	void SetMatrix4x4Array(const unsigned int& AttribLocation, int Count, const void* Data, const unsigned int& Buffer) const;

	//* Appends shader unit to shader program
	void Append(Shader* Shader);

	//* Prepare OpenGL to use this shader
	void Use() const;

	//* Compile shader program
	void Compile();

	//* Unloads the shader program
	void Unload();

	//* The shader is valid for use?
	bool IsValid() const;
};