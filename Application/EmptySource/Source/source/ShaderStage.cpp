#include "../include/Core.h"
#include "../include/GLFunctions.h"
#include "../include/FileManager.h"
#include "../include/ShaderStage.h"

bool ShaderStage::Compile() {
	String Code;
    
    if (ShaderCode == NULL)
        return false;
    
	switch (Type) {
	case Vertex:
		ShaderObject = glCreateShader(GL_VERTEX_SHADER);
		Debug::Log(Debug::LogInfo, L"Compiling VERTEX shader '%ls'", ShaderCode->GetShortPath().c_str());
		Code = WStringToString(FileManager::ReadStream(ShaderCode));
		break;
	case Fragment:
		ShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
		Debug::Log(Debug::LogInfo, L"Compiling FRAGMENT shader '%ls'", ShaderCode->GetShortPath().c_str());
		Code = WStringToString(FileManager::ReadStream(ShaderCode));
		break;
	case Compute:
		// VertexStream = FileManager::Open(ShaderPath);
		break;
	case Geometry:
		ShaderObject = glCreateShader(GL_GEOMETRY_SHADER);
		Debug::Log(Debug::LogInfo, L"Compiling GEOMETRY shader '%ls'", ShaderCode->GetShortPath().c_str());
		Code = WStringToString(FileManager::ReadStream(ShaderCode));
		break;
	}

	const Char * SourcePointer = Code.c_str();
	glShaderSource(ShaderObject, 1, &SourcePointer, NULL);
	glCompileShader(ShaderObject);

	GLint Result = GL_FALSE;

	// --- Check Vertex Object
	glGetShaderiv(ShaderObject, GL_COMPILE_STATUS, &Result);
	if (Result <= 0)
		return false;

	return true;
}

ShaderStage::ShaderStage() {
	ShaderCode = NULL;
	ShaderObject = GL_FALSE;
}

ShaderStage::ShaderStage(ShaderType type, FileStream* ShaderPath) {
	ShaderCode = ShaderPath;
	Type = type;
	Compile();
}

unsigned int ShaderStage::GetShaderObject() const {
	return ShaderObject;
}

ShaderType ShaderStage::GetType() const {
	return Type;
}

void ShaderStage::Delete() {
	glDeleteShader(ShaderObject);
}

bool ShaderStage::IsValid() const {
	return ShaderObject != GL_FALSE;
}
