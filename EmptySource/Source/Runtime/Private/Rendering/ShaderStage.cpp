
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Rendering/GLFunctions.h"
#include "Rendering/ShaderStage.h"
#include "Rendering/Material.h"
#include "Files/FileManager.h"

#include "Utility/TextFormatting.h"

namespace EmptySource {

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

		NString Code;
		if (!ShaderCode->ReadNarrowStream(&Code)) {
			return false;
		}
		CompileFromText(Code);

		ShaderCode->Close();
		return true;
	}

	bool ShaderStage::CompileFromText(const NString & Code) {
		switch (Type) {
		case ST_Vertex:
			ShaderObject = glCreateShader(GL_VERTEX_SHADER);
			LOG_CORE_INFO(L"Compiling VERTEX shader '{:d}'", ShaderObject);
			break;
		case ST_Fragment:
			ShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
			LOG_CORE_INFO(L"Compiling FRAGMENT shader '{:d}'", ShaderObject);
			break;
		case ST_Compute:
			// VertexStream = FileManager::Open(ShaderPath);
			break;
		case ST_Geometry:
			ShaderObject = glCreateShader(GL_GEOMETRY_SHADER);
			LOG_CORE_INFO(L"Compiling GEOMETRY shader '{:d}'", ShaderObject);
			break;
		}

		const NChar * SourcePointer = Code.c_str();
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

}