#include "..\include\Graphics.h"
#include "..\include\Core.h"
#include "..\include\FileManager.h"
#include "..\include\Shader.h"
#include "..\include\ShaderProgram.h"

#include "..\include\Math\Math.h"

bool ShaderProgram::LinkProgram() {
	int InfoLogLength;

	// Link the shader program
	Debug::Log(Debug::LogNormal, L"Linking shader program '%s'...", Name.c_str());
	Program = glCreateProgram();

	if (VertexShader != NULL && VertexShader->IsValid()) {
		glAttachShader(Program, VertexShader->GetShaderUnit());
	}
	if (FragmentShader != NULL && FragmentShader->IsValid()) {
		glAttachShader(Program, FragmentShader->GetShaderUnit());
	}

	glLinkProgram(Program);

	// Check the program
	glGetProgramiv(Program, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		TArray<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(Program, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		Debug::Log(Debug::LogNormal, L"'%s'", CharToWChar((const Char*)&ProgramErrorMessage[0]));
		return false;
	}

	return true;
}

ShaderProgram::ShaderProgram() {
	Shader* VertexShader = NULL;
	Shader* FragmentShader = NULL;
	Shader* ComputeShader = NULL;
	Shader* GeometryShader = NULL;
	WString Name = L"";

	unsigned int Program = GL_FALSE;
}

ShaderProgram::ShaderProgram(WString Name) : ShaderProgram() {
	this->Name = Name;
}

unsigned int ShaderProgram::GetUniformLocation(const Char * LocationName) const {
	return bIsLinked ? glGetUniformLocation(Program, LocationName) : 0;
}

unsigned int ShaderProgram::GetAttribLocation(const Char * LocationName) const {
	return bIsLinked ? glGetAttribLocation(Program, LocationName) : 0;
}

void ShaderProgram::SetMatrix4x4Array(const unsigned int& Location, int Count, const void * Data, const unsigned int & Buffer) const {
	glBindBuffer(GL_ARRAY_BUFFER, Buffer);
	glBufferData(GL_ARRAY_BUFFER, Count * sizeof(Matrix4x4), Data, GL_STATIC_DRAW);

	glEnableVertexAttribArray(Location);
	glVertexAttribPointer(Location, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)0);
	glEnableVertexAttribArray(7);
	glVertexAttribPointer(Location + 1, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(sizeof(Vector4)));
	glEnableVertexAttribArray(8);
	glVertexAttribPointer(Location + 2, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(2 * sizeof(Vector4)));
	glEnableVertexAttribArray(9);
	glVertexAttribPointer(Location + 3, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(3 * sizeof(Vector4)));

	glVertexAttribDivisor(Location    , 1);
	glVertexAttribDivisor(Location + 1, 1);
	glVertexAttribDivisor(Location + 2, 1);
	glVertexAttribDivisor(Location + 3, 1);
}

void ShaderProgram::Append(Shader * shader) {

	if (IsValid()) {
		Debug::Log(Debug::LogError, L"Program '%s' is already linked and compiled, can't modify shader units", Name.c_str());
		return;
	}

	if (!shader->IsValid()) return;

	if (shader->GetType() == Shader::Type::Vertex) {
		VertexShader = shader; return;
	}
	if (shader->GetType() == Shader::Type::Fragment) {
		FragmentShader = shader; return;
	}
	if (shader->GetType() == Shader::Type::Geometry) {
		GeometryShader = shader; return;
	}
	if (shader->GetType() == Shader::Type::Compute) {
		ComputeShader = shader; return;
	}
}

void ShaderProgram::Use() const {
	if (!IsValid()) {
		Debug::Log(Debug::LogError, L"Can't use shader program '%s' because is not valid", Name.c_str());
		return;
	}

	glUseProgram(Program);
}

void ShaderProgram::Compile() {
	bIsLinked = LinkProgram();

	if (bIsLinked == false) {
		glDeleteProgram(Program);
	}
}

void ShaderProgram::Unload() {
	bIsLinked = false;
	glDeleteProgram(Program);
}

bool ShaderProgram::IsValid() const {
	return bIsLinked && Program != GL_FALSE;
}
