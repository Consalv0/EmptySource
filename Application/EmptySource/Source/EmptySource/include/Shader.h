#pragma once

struct FileStream;

class Shader {
private:
	WString FilePath;
	FileStream* VertexStream;
	FileStream* FragmentStream;
	unsigned int VertexShader;
	unsigned int FragmentShader;
	unsigned int ShaderProgram;

	bool bIsLinked;

	//* Create and compile our GLSL shader program from text files
	bool Compile();

	//* Link the shader to OpenGL
	bool LinkProgram();

public:

	Shader();

	/*
	* Create a instance of shader based on the common name of the file in the pathfile specified
	* For example, if the path ../Base this command will search for Base.vertex.glsl 
	* and Base.fragment.glsl files. 
	*/
	Shader(WString FilePath);

	//* Get the location id of a uniform variable in this shader
	unsigned int GetLocationID(const Char* LocationName) const;

	//* Unloads the shader program
	void Unload();

	//* Prepare OpenGL to use this shader
	void Use() const;

	//* The shader is valid for use?
	bool IsValid() const;
};