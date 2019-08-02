#include "../include/Core.h"
#include "../include/FileManager.h"
#include "../include/ShaderStage.h"
#include "../include/Material.h"
#include "../include/GLFunctions.h"

ShaderStage::ShaderStage() {
	ShaderObject = 0;
	Type = ST_Vertex;
}

ShaderStage::ShaderStage(ShaderType type) {
	Type = type;
}

unsigned int ShaderStage::GetShaderObject() const {
	return ShaderObject;
}

ShaderType ShaderStage::GetType() const {
	return Type;
}

bool ShaderStage::CompileFromFile(const WString & FilePath) {
	FileStream * ShaderCode = FileManager::GetFile(FilePath);

	if (ShaderCode == NULL)
		return false;

	CompileFromText(WStringToString(FileManager::ReadStream(ShaderCode)));

	ShaderCode->Close();
}

bool ShaderStage::CompileFromText(const String & Code) {
	switch (Type) {
	case ST_Vertex:
		ShaderObject = glCreateShader(GL_VERTEX_SHADER);
		Debug::Log(Debug::LogInfo, L"Compiling VERTEX shader '%i'", ShaderObject);
		break;
	case ST_Fragment:
		ShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
		Debug::Log(Debug::LogInfo, L"Compiling FRAGMENT shader '%i'", ShaderObject);
		break;
	case ST_Compute:
		// VertexStream = FileManager::Open(ShaderPath);
		break;
	case ST_Geometry:
		ShaderObject = glCreateShader(GL_GEOMETRY_SHADER);
		Debug::Log(Debug::LogInfo, L"Compiling GEOMETRY shader '%i'", ShaderObject);
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

void ShaderStage::Delete() {
	glDeleteShader(ShaderObject);
}

bool ShaderStage::IsValid() const {
	return ShaderObject != GL_FALSE;
}
