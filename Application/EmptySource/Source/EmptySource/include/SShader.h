#pragma once

class SShader {
private:
	std::wstring FilePath;
	std::string VertexShaderCode;
	std::string FragmentShaderCode;
	unsigned int VertexShader;
	unsigned int FragmentShader;
	unsigned int ShaderProgram;

	bool IsLinked;

	//* ReadStreams the file streams of the shader code
	bool ReadStreams(std::fstream* VertexStream, std::fstream* FragmentStream);

	//* Create and compile our GLSL shader program from text files
	bool Compile();

	//* Link the shader to OpenGL
	bool LinkProgram();
public:

	SShader();
	SShader(std::wstring FilePath);

	//* Get the location id of a uniform variable in this shader
	unsigned int GetLocationID(const char* LocationName) const;

	//* Prepare OpenGL to use this shader
	void Use() const;
};