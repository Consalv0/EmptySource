
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Rendering/GLFunctions.h"
#include "Rendering/ShaderProgram.h"
#include "Files/FileManager.h"

namespace EmptySource {

	bool ShaderProgram::LinkProgram() {
		int InfoLogLength;

		// Link the shader program
		LOG_CORE_DEBUG(L"Linking shader program '{}'...", Name);
		ProgramObject = glCreateProgram();

		if (VertexShader != NULL && VertexShader->IsValid()) {
			glAttachShader(ProgramObject, VertexShader->GetShaderObject());
		}
		if (GeometryShader != NULL && GeometryShader->IsValid()) {
			glAttachShader(ProgramObject, GeometryShader->GetShaderObject());
		}
		if (FragmentShader != NULL && FragmentShader->IsValid()) {
			glAttachShader(ProgramObject, FragmentShader->GetShaderObject());
		}

		glLinkProgram(ProgramObject);

		// --- Check the program
		glGetProgramiv(ProgramObject, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			TArray<char> ProgramErrorMessage(InfoLogLength + 1);
			glGetProgramInfoLog(ProgramObject, InfoLogLength, NULL, &ProgramErrorMessage[0]);
			LOG_CORE_ERROR(L"'{}'", Text::NarrowToWide((const NChar*)&ProgramErrorMessage[0]).c_str());
			return false;
		}

		return true;
	}

	ShaderProgram::ShaderProgram() {
		VertexShader = NULL;
		FragmentShader = NULL;
		ComputeShader = NULL;
		GeometryShader = NULL;
		bIsLinked = false;
		WString Name = L"";

		ProgramObject = GL_FALSE;
	}

	ShaderProgram::ShaderProgram(WString Name) : ShaderProgram() {
		VertexShader = NULL;
		FragmentShader = NULL;
		ComputeShader = NULL;
		GeometryShader = NULL;
		bIsLinked = false;
		this->Name = Name;

		ProgramObject = GL_FALSE;
	}

	unsigned int ShaderProgram::GetUniformLocation(const NChar * LocationName) {
		if (bIsLinked == false) return 0;

		auto Finds = Locations.find(LocationName);
		if (Finds != Locations.end()) {
			return Finds->second;
		}

		unsigned int Location = glGetUniformLocation(ProgramObject, LocationName);
		Locations.insert_or_assign(LocationName, Location);
		return Location;
	}

	unsigned int ShaderProgram::GetAttribLocation(const NChar * LocationName) {
		if (bIsLinked == false) return 0;

		auto Finds = Locations.find(LocationName);
		if (Finds != Locations.end()) {
			return Finds->second;
		}

		unsigned int Location = glGetAttribLocation(ProgramObject, LocationName);
		Locations.insert_or_assign(LocationName, Location);
		return Location;
	}

	void ShaderProgram::AppendStage(ShaderStage * Shader) {
		if (IsValid()) {
			LOG_CORE_WARN(L"Program '{}' is already linked and compiled, can't modify shader stages", Name);
			return;
		}

		if (Shader == NULL || !Shader->IsValid()) return;

		if (Shader->GetType() == ST_Vertex) {
			VertexShader = Shader; return;
		}
		if (Shader->GetType() == ST_Fragment) {
			FragmentShader = Shader; return;
		}
		if (Shader->GetType() == ST_Geometry) {
			GeometryShader = Shader; return;
		}
		if (Shader->GetType() == ST_Compute) {
			ComputeShader = Shader; return;
		}
	}

	WString ShaderProgram::GetName() const {
		return Name;
	}

	void ShaderProgram::Use() const {
		if (!IsValid()) {
			LOG_CORE_WARN(L"Can't use shader program '{}' because is not valid", Name);
			return;
		}

		glUseProgram(ProgramObject);
	}

	void ShaderProgram::Compile() {
		bIsLinked = LinkProgram();

		if (bIsLinked == false) {
			glDeleteProgram(ProgramObject);
		}
	}

	void ShaderProgram::Delete() {
		bIsLinked = false;
		glDeleteProgram(ProgramObject);
	}

	bool ShaderProgram::IsValid() const {
		return bIsLinked && ProgramObject != GL_FALSE;
	}

}