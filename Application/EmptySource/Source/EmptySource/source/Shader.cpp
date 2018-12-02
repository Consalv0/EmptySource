#include "..\include\Core.h"
#include "..\include\Shader.h"
#include "..\include\FileManager.h"

bool Shader::Compile() {
	VertexShader = glCreateShader(GL_VERTEX_SHADER);
	FragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Compile Vertex Shader
	wprintf(L"Compiling shader program '%s'\n", FilePath.c_str());
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShader, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShader);

	GLint Result = 0;
	// Check Vertex Shader
	glGetShaderiv(VertexShader, GL_COMPILE_STATUS, &Result);
	if (Result <= 0) {
		return false;
	}

	// Compile Fragment Shader
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShader, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShader);

	// Check Fragment Shader
	glGetShaderiv(FragmentShader, GL_COMPILE_STATUS, &Result);
	if (Result <= 0) {
		return false;
	}

	return true;
}

bool Shader::ReadStreams(std::fstream * VertexStream, std::fstream * FragmentStream) {
	// ReadStreams the Vertex Shader code from the file
	if (VertexStream != NULL && VertexStream->is_open()) {
		std::stringstream sstr;
		try {
			sstr << VertexStream->rdbuf();
		} catch (...) {
			return false;
		}
		VertexShaderCode = sstr.str();
		VertexStream->close();
	} else {
		wprintf(L"Impossible to open \"%s\". Are you in the right directory ?\n", FileManager::GetFilePath(VertexStream).c_str());
		return false;
	}

	// ReadStreams the Fragment Shader code from the file
	if (FragmentStream != NULL && FragmentStream->is_open()) {
		std::stringstream sstr;
		try {
			sstr << FragmentStream->rdbuf();
		} catch (...) {
			return false;
		}
		FragmentShaderCode = sstr.str();
		FragmentStream->close();
	} else {
		wprintf(L"Impossible to open \"%s\". Are you in the right directory ?\n", FileManager::GetFilePath(FragmentStream).c_str());
		return false;
	}

	return true;
}

bool Shader::LinkProgram() {
	int InfoLogLength;

	// Link the shader program
	wprintf(L"└>Linking shader program '%s'\n", FilePath.c_str());
	ShaderProgram = glCreateProgram();
	glAttachShader(ShaderProgram, VertexShader);
	glAttachShader(ShaderProgram, FragmentShader);
	glLinkProgram(ShaderProgram);

	// Check the program
	glGetProgramiv(ShaderProgram, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ShaderProgram, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		wprintf(L"%s\n", FChar((const char*)&ProgramErrorMessage[0]));
		return false;
	}

	return true;
}

Shader::Shader() {
	IsLinked = false;
	VertexShader = 0;
	FragmentShader = 0; 
	ShaderProgram = 0;
	FilePath = L"";
}

Shader::Shader(std::wstring ShaderNamePath) {
	FilePath = ShaderNamePath;

	IsLinked = ReadStreams(
		FileManager::Open(FilePath + L".vertex.glsl"),
		FileManager::Open(FilePath + L".fragment.glsl")
	);
	if (IsLinked == false) return;

	IsLinked &= Compile();
	IsLinked &= LinkProgram();

	if (IsLinked == false) {
		glDeleteProgram(ShaderProgram);
		glDetachShader(ShaderProgram, VertexShader);
		glDetachShader(ShaderProgram, FragmentShader);

		glDeleteShader(VertexShader);
		glDeleteShader(FragmentShader);
	}
}

unsigned int Shader::GetLocationID(const char * LocationName) const {
	return IsLinked ? glGetUniformLocation(ShaderProgram, LocationName) : 0;
}

void Shader::Use() const {
	if (IsLinked) glUseProgram(ShaderProgram);
}
