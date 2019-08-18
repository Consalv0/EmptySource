
#include "CoreMinimal.h"
#include "Rendering/GLFunctions.h"
#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Math/MathUtility.h"
#include "Math/Vector4.h"
#include "Math/Quaternion.h"
#include "Math/Matrix4x4.h"

namespace EmptySource {

	Material::Material() {
		MaterialShader = NULL;
		RenderPriority = 1000;
		bUseDepthTest = true;
		DepthFunction = DF_LessEqual;
		FillMode = FM_Solid;
		CullMode = CM_CounterClockWise;
	}

	void Material::SetShaderProgram(ShaderPtr Value) {
		if (Value != NULL && Value->IsValid()) {
			MaterialShader = Value;
		}
		else {
			LOG_CORE_ERROR(L"The Shader Program '{}' is not a valid program", Value != NULL ? Value->GetName() : L"NULL");
		}
	}

	ShaderPtr Material::GetShaderProgram() const {
		return MaterialShader;
	}

	void Material::SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void * Data, const unsigned int & Buffer) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int AttribLocation = Program->GetAttribLocation(AttributeName);

		glBindBuffer(GL_ARRAY_BUFFER, Buffer);
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

	void Material::SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniformMatrix4fv(UniformLocation, Count, GL_FALSE, Data);
	}

	void Material::SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform1fv(UniformLocation, Count, Data);
	}

	void Material::SetInt1Array(const NChar * UniformName, const int * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform1iv(UniformLocation, Count, (GLint *)Data);
	}

	void Material::SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform2fv(UniformLocation, Count, Data);
	}

	void Material::SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform3fv(UniformLocation, Count, Data);
	}

	void Material::SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform4fv(UniformLocation, Count, Data);
	}

	void Material::SetTexture2D(const NChar * UniformName, TexturePtr Tex, const unsigned int & Position) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL || Tex == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform1i(UniformLocation, Position);
		glActiveTexture(GL_TEXTURE0 + Position);
		Tex->Bind();
	}

	void Material::SetTextureCubemap(const NChar * UniformName, TexturePtr Tex, const unsigned int & Position) const {
		ShaderPtr Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform1i(UniformLocation, Position);
		glActiveTexture(GL_TEXTURE0 + Position);
		Tex->Bind();
	}

	void Material::Use() const {
		// --- Activate Z-buffer
		if (bUseDepthTest) {
			glEnable(GL_DEPTH_TEST);
		}
		else {
			glDisable(GL_DEPTH_TEST);
		}

		switch (DepthFunction) {
		case DF_Always:
			glDepthFunc(GL_ALWAYS); break;
		case DF_Equal:
			glDepthFunc(GL_EQUAL); break;
		case DF_Greater:
			glDepthFunc(GL_GREATER); break;
		case DF_GreaterEqual:
			glDepthFunc(GL_GEQUAL); break;
		case DF_Less:
			glDepthFunc(GL_LESS); break;
		case DF_LessEqual:
			glDepthFunc(GL_LEQUAL); break;
		case DF_Never:
			glDepthFunc(GL_NEVER); break;
		case DF_NotEqual:
			glDepthFunc(GL_NOTEQUAL); break;
		}

		if (CullMode == CM_None) {
			glDisable(GL_CULL_FACE);
		}
		else {
			glEnable(GL_CULL_FACE);
			switch (CullMode) {
			case CM_ClockWise:
				glCullFace(GL_FRONT); break;
			case CM_CounterClockWise:
				glCullFace(GL_BACK); break;
			case CM_None:
				break;
			}
		}

		switch (FillMode) {
		case FM_Point:
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); break;
		case FM_Wireframe:
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); break;
		case FM_Solid:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); break;
		}

		if (MaterialShader && MaterialShader->IsValid()) {
			MaterialShader->Bind();
		}
	}

}