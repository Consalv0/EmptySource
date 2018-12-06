#include "..\include\Core.h"
#include "..\include\Shader.h"
#include "..\include\FileManager.h"

bool Shader::Compile() {
	VertexShader = glCreateShader(GL_VERTEX_SHADER);
	FragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	_LOG(Log, L"Compiling shader program '%s'", FilePath.c_str());

	// Compile Vertex Shader
	String VertexShaderCode = ToString(FileManager::ReadStream(VertexStream));
	const Char * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShader, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShader);

	GLint Result = 0;
	// Check Vertex Shader
	glGetShaderiv(VertexShader, GL_COMPILE_STATUS, &Result);
	if (Result <= 0) {
		return false;
	}

	// Compile Fragment Shader
	String FragmentShaderCode = ToString(FileManager::ReadStream(FragmentStream));
	const Char * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShader, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShader);

	// Check Fragment Shader
	glGetShaderiv(FragmentShader, GL_COMPILE_STATUS, &Result);
	if (Result <= 0) {
		return false;
	}

	return true;
}

bool Shader::LinkProgram() {
	int InfoLogLength;

	// Link the shader program
	_LOG(Log, L"└>Linking shader program '%s'", FilePath.c_str());
	ShaderProgram = glCreateProgram();
	glAttachShader(ShaderProgram, VertexShader);
	glAttachShader(ShaderProgram, FragmentShader);
	glLinkProgram(ShaderProgram);

	// Check the program
	glGetProgramiv(ShaderProgram, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ShaderProgram, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		_LOG(NoLog, L"'%s'", ToWChar((const char*)&ProgramErrorMessage[0]));
		return false;
	}

	return true;
}

Shader::Shader() {
	bIsLinked = false;
	VertexShader = 0;
	FragmentShader = 0; 
	ShaderProgram = 0;
	FilePath = L"";
}

Shader::Shader(WString ShaderNamePath) {
	FilePath = ShaderNamePath;

	VertexStream = FileManager::Open(FilePath + L".vertex.glsl");
	FragmentStream = FileManager::Open(FilePath + L".fragment.glsl");

	if (VertexStream == NULL || FragmentStream == NULL) return;

	Compile();
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
	if (IsValid()) glUseProgram(ShaderProgram);
}

bool Shader::IsValid() const {
	return bIsLinked;
}
