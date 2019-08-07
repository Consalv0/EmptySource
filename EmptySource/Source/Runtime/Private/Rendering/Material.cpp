
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Rendering/GLFunctions.h"
#include "Rendering/Material.h"
#include "Rendering/Texture2D.h"
#include "Rendering/Cubemap.h"
#include "Math/MathUtility.h"
#include "Math/Vector4.h"
#include "Math/Quaternion.h"
#include "Math/Matrix4x4.h"

namespace EmptySource {

	Material::Material() {
		MaterialShader = NULL;
		RenderPriority = 1000;
		bUseDepthTest = true;
		DepthFunction = Graphics::DF_LessEqual;
		RenderMode = Graphics::RM_Fill;
		CullMode = Graphics::CM_Back;
	}

	void Material::SetShaderProgram(ShaderProgram* Value) {
		if (Value != NULL && Value->IsValid()) {
			MaterialShader = Value;
		}
		else {
			LOG_CORE_ERROR(L"The Shader Program '{}' is not a valid program", Value != NULL ? Value->GetName() : L"NULL");
		}
	}

	ShaderProgram * Material::GetShaderProgram() const {
		return MaterialShader;
	}

	void Material::SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void * Data, const unsigned int & Buffer) const {
		ShaderProgram * Program = GetShaderProgram();
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
		ShaderProgram * Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniformMatrix4fv(UniformLocation, Count, GL_FALSE, Data);
	}

	void Material::SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderProgram * Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform1fv(UniformLocation, Count, Data);
	}

	void Material::SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderProgram * Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform2fv(UniformLocation, Count, Data);
	}

	void Material::SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderProgram * Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform3fv(UniformLocation, Count, Data);
	}

	void Material::SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count) const {
		ShaderProgram * Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform4fv(UniformLocation, Count, Data);
	}

	void Material::SetTexture2D(const NChar * UniformName, Texture2D * Tex, const unsigned int & Position) const {
		ShaderProgram * Program = GetShaderProgram();
		if (Program == NULL || Tex == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform1i(UniformLocation, Position);
		glActiveTexture(GL_TEXTURE0 + Position);
		Tex->Use();
	}

	void Material::SetTextureCubemap(const NChar * UniformName, Cubemap * Tex, const unsigned int & Position) const {
		ShaderProgram * Program = GetShaderProgram();
		if (Program == NULL) return;
		unsigned int UniformLocation = Program->GetUniformLocation(UniformName);
		glUniform1i(UniformLocation, Position);
		glActiveTexture(GL_TEXTURE0 + Position);
		Tex->Use();
	}

	void Material::Use() {
		// --- Activate Z-buffer
		if (bUseDepthTest) {
			glEnable(GL_DEPTH_TEST);
		}
		else {
			glDisable(GL_DEPTH_TEST);
		}

		switch (DepthFunction) {
		case Graphics::DF_Always:
			glDepthFunc(GL_ALWAYS); break;
		case Graphics::DF_Equal:
			glDepthFunc(GL_EQUAL); break;
		case Graphics::DF_Greater:
			glDepthFunc(GL_GREATER); break;
		case Graphics::DF_GreaterEqual:
			glDepthFunc(GL_GEQUAL); break;
		case Graphics::DF_Less:
			glDepthFunc(GL_LESS); break;
		case Graphics::DF_LessEqual:
			glDepthFunc(GL_LEQUAL); break;
		case Graphics::DF_Never:
			glDepthFunc(GL_NEVER); break;
		case Graphics::DF_NotEqual:
			glDepthFunc(GL_NOTEQUAL); break;
		}

		if (CullMode == Graphics::CM_None) {
			glDisable(GL_CULL_FACE);
		}
		else {
			glEnable(GL_CULL_FACE);
			switch (CullMode) {
			case Graphics::CM_Front:
				glCullFace(GL_FRONT); break;
			case Graphics::CM_Back:
				glCullFace(GL_BACK); break;
			case Graphics::CM_FrontBack:
				glCullFace(GL_FRONT_AND_BACK); break;
			case Graphics::CM_None:
				break;
			}
		}

		switch (RenderMode) {
		case Graphics::RM_Point:
			glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); break;
		case Graphics::RM_Wire:
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); break;
		case Graphics::RM_Fill:
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); break;
		}

		if (MaterialShader && MaterialShader->IsValid()) {
			MaterialShader->Use();
		}
	}

}