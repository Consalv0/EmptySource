#pragma once

#include "..\include\ShaderStage.h"
#include <functional>
#include <cstring>

typedef TDictionary<const Char *, unsigned int> LocationMap;

class ShaderProgram {
private:
	ShaderStage* VertexShader;
	ShaderStage* FragmentShader;
	ShaderStage* ComputeShader;
	ShaderStage* GeometryShader;
	WString Name;

	bool bIsLinked;

	//* Link the shader to OpenGL
	bool LinkProgram();

	//* Shader Program Object
	unsigned int ProgramObject;

	LocationMap Locations;

public:
	//* Default Constructor
	ShaderProgram();

	//* Constructor with name
	ShaderProgram(WString Name);

	//* Get the location id of a uniform variable in this shader
	unsigned int GetUniformLocation(const Char* LocationName);

	//* Get the location of the attrib in this shader
	unsigned int GetAttribLocation(const Char* LocationName);

	//* Appends shader unit to shader program
	void AppendStage(ShaderStage* ShaderProgram);

	//* Get the string name of this program
	WString GetName() const;

	//* Prepare OpenGL to use this shader
	void Use() const;

	//* Compile shader program
	void Compile();

	//* Unloads the shader program
	void Delete();

	//* The shader is valid for use?
	bool IsValid() const;
};