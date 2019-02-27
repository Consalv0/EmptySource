#include "..\include\Core.h"
#include "..\include\CoreGraphics.h"
#include "..\include\FileManager.h"
#include "..\include\ShaderProgram.h"

#include "..\include\Math\Math.h"

bool ShaderProgram::LinkProgram() {
	int InfoLogLength;

	// Link the shader program
	Debug::Log(Debug::LogNormal, L"Linking shader program '%s'...", Name.c_str());
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

	// Check the program
	glGetProgramiv(ProgramObject, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		TArray<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramObject, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		Debug::Log(Debug::LogNormal, L"'%s'", CharToWChar((const Char*)&ProgramErrorMessage[0]));
		return false;
	}

	return true;
}

ShaderProgram::ShaderProgram() {
	ShaderStage* VertexShader = NULL;
	ShaderStage* FragmentShader = NULL;
	ShaderStage* ComputeShader = NULL;
	ShaderStage* GeometryShader = NULL;
	WString Name = L"";

	unsigned int Program = GL_FALSE;
}

ShaderProgram::ShaderProgram(WString Name) : ShaderProgram() {
	this->Name = Name;
}

unsigned int ShaderProgram::GetUniformLocation(const Char * LocationName) {
	if (bIsLinked == false) return 0;

	auto Finds = Locations.find(LocationName);
	if (Finds != Locations.end()) {
		return Finds->second;
	}

	unsigned int Location = glGetUniformLocation(ProgramObject, LocationName);
	Locations.insert_or_assign(LocationName, Location);
	return Location;
}

unsigned int ShaderProgram::GetAttribLocation(const Char * LocationName) {
	if (bIsLinked == false) return 0;

	auto Finds = Locations.find(LocationName);
	if (Finds != Locations.end()) {
		return Finds->second;
	}

	unsigned int Location = glGetAttribLocation(ProgramObject, LocationName);
	Locations.insert_or_assign(LocationName, Location);
	return Location;
}

void ShaderProgram::AppendStage(ShaderStage * shader) {

	if (IsValid()) {
		Debug::Log(Debug::LogError, L"Program '%s' is already linked and compiled, can't modify shader stages", Name.c_str());
		return;
	}

	if (!shader->IsValid()) return;

	if (shader->GetType() == ShaderType::Vertex) {
		VertexShader = shader; return;
	}
	if (shader->GetType() == ShaderType::Fragment) {
		FragmentShader = shader; return;
	}
	if (shader->GetType() == ShaderType::Geometry) {
		GeometryShader = shader; return;
	}
	if (shader->GetType() == ShaderType::Compute) {
		ComputeShader = shader; return;
	}
}

WString ShaderProgram::GetName() const {
	return Name;
}

void ShaderProgram::Use() const {
	if (!IsValid()) {
		Debug::Log(Debug::LogError, L"Can't use shader program '%s' because is not valid", Name.c_str());
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
