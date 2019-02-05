#include "..\include\Graphics.h"
#include "..\include\Core.h"
#include "..\include\FileManager.h"
#include "..\include\Shader.h"

bool Shader::Compile() {
	String Code;

	switch (ShaderType) {
	case Vertex:
		ShaderUnit = glCreateShader(GL_VERTEX_SHADER);
		Debug::Log(Debug::LogNormal, L"Compiling vertex shader '%s'", ShaderCode->GetShortPath().c_str());
		Code = WStringToString(FileManager::ReadStream(ShaderCode));
		break;
	case Fragment:
		ShaderUnit = glCreateShader(GL_FRAGMENT_SHADER);
		Debug::Log(Debug::LogNormal, L"Compiling fragment shader '%s'", ShaderCode->GetShortPath().c_str());
		Code = WStringToString(FileManager::ReadStream(ShaderCode));
		break;
	case Compute:
		// VertexStream = FileManager::Open(ShaderPath);
		break;
	case Geometry:
		ShaderUnit = glCreateShader(GL_GEOMETRY_SHADER);
		Debug::Log(Debug::LogNormal, L"Compiling geometry shader '%s'", ShaderCode->GetShortPath().c_str());
		Code = WStringToString(FileManager::ReadStream(ShaderCode));
		break;
	}

	const Char * SourcePointer = Code.c_str();
	glShaderSource(ShaderUnit, 1, &SourcePointer, NULL);
	glCompileShader(ShaderUnit);

	GLint Result = GL_FALSE;
	// Check Vertex Shader
	glGetShaderiv(ShaderUnit, GL_COMPILE_STATUS, &Result);
	if (Result <= 0) {
		return false;
	}

	return true;
}

Shader::Shader() {
	ShaderCode = NULL;
	ShaderUnit = GL_FALSE;
}

Shader::Shader(Type type, FileStream* ShaderPath) {
	ShaderCode = ShaderPath;
	ShaderType = type;
	Compile();
}

unsigned int Shader::GetShaderUnit() const {
	return ShaderUnit;
}

Shader::Type Shader::GetType() const {
	return ShaderType;
}

void Shader::Unload() {
	glDeleteShader(ShaderUnit);
}

bool Shader::IsValid() const {
	return ShaderUnit != GL_FALSE;
}
