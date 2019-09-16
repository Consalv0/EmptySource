
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Shader.h"
#include "Rendering/RenderingAPI.h"

#include "Platform/OpenGL/OpenGLShader.h"
#include "Platform/OpenGL/OpenGLAPI.h"

#include "Files/FileManager.h"

#include "glad/glad.h"

namespace EmptySource {

	OpenGLShaderStage::OpenGLShaderStage(const WString & FilePath, EShaderType Type) : StageType(Type) {
		bValid = false;
		ShaderObject = GL_FALSE;
		FileStream * ShaderCode = FileManager::GetFile(FilePath);

		if (ShaderCode == NULL) {
			LOG_CORE_ERROR(L"Error reading file for shader: '{}'", FilePath);
			return;
		}

		NString Code;
		if (!ShaderCode->ReadNarrowStream(&Code)) {
			ShaderCode->Close();
			LOG_CORE_ERROR(L"Error reading file for shader: '{}'", FilePath);
			return;
		}

		switch (Type) {
		case ST_Vertex:
			ShaderObject = glCreateShader(GL_VERTEX_SHADER);
			LOG_CORE_INFO(L"Compiling VERTEX shader '{:d}'", ShaderObject);
			break;
		case ST_Pixel:
			ShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
			LOG_CORE_INFO(L"Compiling PIXEL shader '{:d}'", ShaderObject);
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

		GLint LogSize = 0;
		glGetShaderiv(ShaderObject, GL_INFO_LOG_LENGTH, &LogSize);
		if (LogSize > 0) {
			LOG_CORE_ERROR(L"Error while compiling shader '{:d}'", ShaderObject);
			TArray<NChar> ShaderErrorMessage(LogSize + 1);
			glGetShaderInfoLog(ShaderObject, LogSize, NULL, &ShaderErrorMessage[0]);
			LOG_CORE_ERROR("'{}'", NString((const NChar*)&ShaderErrorMessage[0]));
		}
		else
			bValid = true;

		ShaderCode->Close();
	}

	OpenGLShaderStage::OpenGLShaderStage(const NString & Code, EShaderType Type) : StageType(Type) {
		bValid = false;
		ShaderObject = GL_FALSE;

		switch (Type) {
		case ST_Vertex:
			ShaderObject = glCreateShader(GL_VERTEX_SHADER);
			LOG_CORE_INFO(L"Compiling VERTEX shader '{:d}'", ShaderObject);
			break;
		case ST_Pixel:
			ShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
			LOG_CORE_INFO(L"Compiling PIXEL shader '{:d}'", ShaderObject);
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

		GLint LogSize = 0;
		glGetShaderiv(ShaderObject, GL_INFO_LOG_LENGTH, &LogSize);
		if (LogSize > 0) {
			LOG_CORE_ERROR(L"Error while compiling shader '{:d}'", ShaderObject);
			TArray<NChar> ShaderErrorMessage(LogSize + 1);
			glGetShaderInfoLog(ShaderObject, LogSize, NULL, &ShaderErrorMessage[0]);
			LOG_CORE_ERROR("'{}'", NString((const NChar*)&ShaderErrorMessage[0]));
		} else
			bValid = true;
	}

	OpenGLShaderStage::~OpenGLShaderStage() {
		LOG_CORE_DEBUG(L"Deleting shader stage '{:d}'...", ShaderObject);
		glDeleteProgram(ShaderObject);
	}

	// Shader Program

	OpenGLShaderProgram::OpenGLShaderProgram(const WString& Name, TArray<ShaderStagePtr> ShaderStages) 
		: Name(Name), Locations(), Properties()
	{
		bValid = false;
		VertexShader = NULL;
		PixelShader = NULL;
		ComputeShader = NULL;
		GeometryShader = NULL;
		ProgramObject = GL_FALSE;
		
		for (auto Shader : ShaderStages)
			AppendStage(Shader);

		int InfoLogLength;

		// Link the shader program
		LOG_CORE_DEBUG(L"Linking shader program '{}'...", Name);
		ProgramObject = glCreateProgram();

		if (VertexShader != NULL && VertexShader->IsValid()) {
			glAttachShader(ProgramObject, (unsigned int)(unsigned long long)VertexShader->GetStageObject());
		}
		if (GeometryShader != NULL && GeometryShader->IsValid()) {
			glAttachShader(ProgramObject, (unsigned int)(unsigned long long)GeometryShader->GetStageObject());
		}
		if (PixelShader != NULL && PixelShader->IsValid()) {
			glAttachShader(ProgramObject, (unsigned int)(unsigned long long)PixelShader->GetStageObject());
		}

		glLinkProgram(ProgramObject);

		// Check the program
		glGetProgramiv(ProgramObject, GL_INFO_LOG_LENGTH, &InfoLogLength);
		if (InfoLogLength > 0) {
			TArray<NChar> ProgramErrorMessage(InfoLogLength + 1);
			glGetProgramInfoLog(ProgramObject, InfoLogLength, NULL, &ProgramErrorMessage[0]);
			LOG_CORE_ERROR("'{}'", NString((const NChar*)&ProgramErrorMessage[0]));
		}
		else
			bValid = true;
	}

	OpenGLShaderProgram::~OpenGLShaderProgram() {
		LOG_CORE_DEBUG(L"Deleting shader program '{}'...", Name);
		glDeleteProgram(ProgramObject);
	}

	void OpenGLShaderProgram::Bind() const {
		if (!IsValid()) LOG_CORE_WARN(L"Can't use shader program '{}' because is not valid", Name);
		glUseProgram(ProgramObject);
	}

	void OpenGLShaderProgram::Unbind() const {
		glUseProgram(0);
	}

	unsigned int OpenGLShaderProgram::GetUniformLocation(const NChar * LocationName) {
		if (!IsValid()) return 0;

		auto Find = Locations.find(LocationName);
		if (Find != Locations.end()) {
			return Find->second;
		}

		unsigned int Location = glGetUniformLocation(ProgramObject, LocationName);
		if (Location == -1) {
			LOG_CORE_WARN("Setting variable to uniform location not present in {0} : {1}",
				Text::WideToNarrow(GetName()).c_str(), LocationName);
		}
		Locations.emplace(LocationName, Location);
		return Location;
	}

	unsigned int OpenGLShaderProgram::GetAttribLocation(const NChar * LocationName) {
		if (!IsValid()) return 0;

		auto Finds = Locations.find(LocationName);
		if (Finds != Locations.end()) {
			return Finds->second;
		}

		unsigned int Location = glGetAttribLocation(ProgramObject, LocationName);
		if (Location == -1) {
			LOG_CORE_WARN("Setting variable to attribute location not present in {0} : {1}",
				Text::WideToNarrow(GetName()).c_str(), LocationName);
		}
		Locations.insert_or_assign(LocationName, Location);
		return Location;
	}

	void OpenGLShaderProgram::SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void * Data, const VertexBufferPtr & Buffer) {
		if (!IsValid()) return;
		unsigned int AttribLocation = GetAttribLocation(AttributeName);

		Buffer->Bind();
		glBufferData(GL_ARRAY_BUFFER, Count * sizeof(Matrix4x4), Data, GL_STATIC_DRAW);

		glEnableVertexAttribArray(AttribLocation);
		glVertexAttribPointer(AttribLocation, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)0);
		glEnableVertexAttribArray(7);
		glVertexAttribPointer(AttribLocation + 1, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(sizeof(Vector4)));
		glEnableVertexAttribArray(8);
		glVertexAttribPointer(AttribLocation + 2, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(2 * sizeof(Vector4)));
		glEnableVertexAttribArray(9);
		glVertexAttribPointer(AttribLocation + 3, 4, GL_FLOAT, GL_FALSE, sizeof(Matrix4x4), (void*)(3 * sizeof(Vector4)));

		glVertexAttribDivisor(AttribLocation, 1);
		glVertexAttribDivisor(AttribLocation + 1, 1);
		glVertexAttribDivisor(AttribLocation + 2, 1);
		glVertexAttribDivisor(AttribLocation + 3, 1);
	}

	void OpenGLShaderProgram::SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count) {
		if (!IsValid()) return;
		unsigned int UniformLocation = GetUniformLocation(UniformName);
		glUniformMatrix4fv(UniformLocation, Count, GL_FALSE, Data);
	}

	void OpenGLShaderProgram::SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count) {
		if (!IsValid()) return;
		unsigned int UniformLocation = GetUniformLocation(UniformName);
		glUniform1fv(UniformLocation, Count, Data);
	}

	void OpenGLShaderProgram::SetInt1Array(const NChar * UniformName, const int * Data, const int & Count) {
		if (!IsValid()) return;
		unsigned int UniformLocation = GetUniformLocation(UniformName);
		glUniform1iv(UniformLocation, Count, (GLint *)Data);
	}

	void OpenGLShaderProgram::SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count) {
		if (!IsValid()) return;
		unsigned int UniformLocation = GetUniformLocation(UniformName);
		glUniform2fv(UniformLocation, Count, Data);
	}

	void OpenGLShaderProgram::SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count) {
		if (!IsValid()) return;
		unsigned int UniformLocation = GetUniformLocation(UniformName);
		glUniform3fv(UniformLocation, Count, Data);
	}

	void OpenGLShaderProgram::SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count) {
		if (!IsValid()) return;
		unsigned int UniformLocation = GetUniformLocation(UniformName);
		glUniform4fv(UniformLocation, Count, Data);
	}

	void OpenGLShaderProgram::SetTexture2D(const NChar * UniformName, TexturePtr Tex, const unsigned int & Position) {
		if (!IsValid() || Tex == NULL) return;
		unsigned int UniformLocation = GetUniformLocation(UniformName);
		glUniform1i(UniformLocation, Position);
		glActiveTexture(GL_TEXTURE0 + Position);
		Tex->Bind();
	}

	void OpenGLShaderProgram::SetTextureCubemap(const NChar * UniformName, TexturePtr Tex, const unsigned int & Position) {
		if (!IsValid() || Tex == NULL) return;
		unsigned int UniformLocation = GetUniformLocation(UniformName);
		glUniform1i(UniformLocation, Position);
		glActiveTexture(GL_TEXTURE0 + Position);
		Tex->Bind();
	}

	void OpenGLShaderProgram::SetProperties(const TArrayInitializer<ShaderProperty> & Properties) {
		this->Properties = Properties;
	}

	void OpenGLShaderProgram::SetProperties(const TArray<ShaderProperty>& Properties) {
		this->Properties = Properties;
	}

	void OpenGLShaderProgram::AppendStage(ShaderStagePtr Shader) {
		if (IsValid()) {
			LOG_CORE_WARN(L"Program '{}' is already linked and compiled, can't modify shader stages", Name);
			return;
		}

		if (Shader == NULL || !Shader->IsValid()) return;

		switch (Shader->GetType()) {
			case ST_Vertex:
				VertexShader = Shader; return;
			case ST_Pixel:
				PixelShader = Shader; return;
			case ST_Geometry:
				GeometryShader = Shader; return;
			case ST_Compute:
				ComputeShader = Shader; return;
			default:
				break;
		}
	}

}