#include "..\include\Graphics.h"
#include "..\include\Core.h"
#include "..\include\FileManager.h"
#include "..\include\Shader.h"

#include "..\include\Math\Math.h"

bool Shader::Compile(ShaderType Type) {

	// Compile Vertex Shader
	String ShaderCode;
	unsigned* ShaderID = NULL;

	switch (Type) {
	case Vertex:
		VertexShader = glCreateShader(GL_VERTEX_SHADER);
		Debug::Log(Debug::LogNormal, L"Compiling vertex shader '%s'.vertex.glsl", VertexStream->GetShortPath().c_str());
		ShaderCode = WStringToString(FileManager::ReadStream(VertexStream));
		ShaderID = &VertexShader;
		break;
	case Fragment:
		FragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		Debug::Log(Debug::LogNormal, L"Compiling fragment shader '%s'.fragment.glsl", FragmentStream->GetShortPath().c_str());
		ShaderCode = WStringToString(FileManager::ReadStream(FragmentStream));
		ShaderID = &FragmentShader;
		break;
	case Pixel:
		// VertexStream = FileManager::Open(ShaderPath + L".pixel.glsl");
		break;
	case Geometry:
		// VertexStream = FileManager::Open(ShaderPath + L".geometry.glsl");
		break;
	}

	const Char * SourcePointer = ShaderCode.c_str();
	glShaderSource(*ShaderID, 1, &SourcePointer, NULL);
	glCompileShader(*ShaderID);

	GLint Result = GL_FALSE;
	// Check Vertex Shader
	glGetShaderiv(*ShaderID, GL_COMPILE_STATUS, &Result);
	if (Result <= 0) {
		return false;
	}

	return true;
}

bool Shader::LinkProgram() {
	int InfoLogLength;

	// Link the shader program
	Debug::Log(Debug::LogNormal, L"└> Linking shader program '%s'", Name.c_str());
	ShaderProgram = glCreateProgram();

	if (VertexShader != GL_FALSE)
	glAttachShader(ShaderProgram, VertexShader);
	if (FragmentShader != GL_FALSE)
	glAttachShader(ShaderProgram, FragmentShader);

	glLinkProgram(ShaderProgram);

	// Check the program
	glGetProgramiv(ShaderProgram, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		TArray<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ShaderProgram, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		Debug::Log(Debug::LogNormal, L"'%s'", CharToWChar((const Char*)&ProgramErrorMessage[0]));
		return false;
	}

	return true;
}

Shader::Shader() {
	bIsLinked = false;
	VertexShader = GL_FALSE;
	FragmentShader = GL_FALSE;
	ShaderProgram = GL_FALSE;
	VertexStream = NULL;
	FragmentStream = NULL;
	Name = L"";
}

Shader::Shader(const WString & name) {
	Name = name;
}

void Shader::LoadShader(ShaderType Type, WString ShaderPath) {
	switch (Type) {
	case Vertex:
		VertexStream = FileManager::Open(ShaderPath + L".vertex.glsl");
		if (VertexStream == NULL) return;
		break;
	case Fragment:
		FragmentStream = FileManager::Open(ShaderPath + L".fragment.glsl");
		if (FragmentStream == NULL) return;
		break;
	case Pixel:
		// VertexStream = FileManager::Open(ShaderPath + L".pixel.glsl");
		break;
	case Geometry:
		// VertexStream = FileManager::Open(ShaderPath + L".geometry.glsl");
		break;
	}

	Compile(Type);
}

void Shader::Compile() {
	bIsLinked = LinkProgram();

	if (bIsLinked == false) {
		glDeleteProgram(ShaderProgram);
		glDetachShader(ShaderProgram, VertexShader);
		glDetachShader(ShaderProgram, FragmentShader);

		glDeleteShader(VertexShader);
		glDeleteShader(FragmentShader);
	}
}

unsigned int Shader::GetLocationID(const Char * LocationName) const {
	return bIsLinked ? glGetUniformLocation(ShaderProgram, LocationName) : 0;
}

void Shader::Unload() {
	bIsLinked = false;
	glDeleteShader(VertexShader);
	glDeleteShader(FragmentShader);
	glDeleteProgram(ShaderProgram);
}

void Shader::Use() const {
	if (!IsValid()) {
		Debug::Log(Debug::LogError, L"Can't use shader '%s' because is not valid", Name.c_str());
		return;
	}

	glUseProgram(ShaderProgram);
}

bool Shader::IsValid() const {
	return bIsLinked && ShaderProgram != GL_FALSE;
}
