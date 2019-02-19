#include "..\include\Core.h"
#include "..\include\Material.h"
#include "..\include\Math\Math.h"

Material::Material() {
	MaterialShader = NULL;
	bUseDepthTest = true;
	DepthFunction = Graphics::DepthFunction::LessEqual;
	RenderMode = Graphics::RenderMode::Fill;
	CullMode = Graphics::CullMode::Back;
}

void Material::SetShaderProgram(ShaderProgram* Value) {
	if (Value != NULL && Value->IsValid()) {
		MaterialShader = Value;
	} else {
		Debug::Log(
			Debug::LogError, L"The Shader Program '%s' is not a valid program",
			Value != NULL ? Value->GetName().c_str() : L"NULL"
		);
	}
}

ShaderProgram * Material::GetShaderProgram() const {
	return MaterialShader;
}

void Material::SetAttribMatrix4x4Array(const Char * AttributeName, int Count, const void * Data, const unsigned int & Buffer) const {
	unsigned int AttribLocation = GetShaderProgram()->GetAttribLocation(AttributeName);

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

void Material::SetMatrix4x4Array(const Char * UniformName, const float * Data, const int & Count) const {
	unsigned int UniformLocation = GetShaderProgram()->GetUniformLocation(UniformName);
	glUniformMatrix4fv(UniformLocation, Count, GL_FALSE, Data);
}

void Material::SetFloat1Array(const Char * UniformName, const float * Data, const int & Count) const {
	unsigned int UniformLocation = GetShaderProgram()->GetUniformLocation(UniformName);
	glUniform1fv(UniformLocation, Count, Data);
}

void Material::SetFloat2Array(const Char * UniformName, const float * Data, const int & Count) const {
	unsigned int UniformLocation = GetShaderProgram()->GetUniformLocation(UniformName);
	glUniform2fv(UniformLocation, Count, Data);
}

void Material::SetFloat3Array(const Char * UniformName, const float * Data, const int & Count) const {
	unsigned int UniformLocation = GetShaderProgram()->GetUniformLocation(UniformName);
	glUniform3fv(UniformLocation, Count, Data);
}

void Material::SetFloat4Array(const Char * UniformName, const float * Data, const int & Count) const {
	unsigned int UniformLocation = GetShaderProgram()->GetUniformLocation(UniformName);
	glUniform4fv(UniformLocation, Count, Data);
}

void Material::Use() {
	// --- Activate Z-buffer
	if (bUseDepthTest) {
		glEnable(GL_DEPTH_TEST);
	} else {
		glDisable(GL_DEPTH_TEST);
	}

	glDepthFunc((unsigned int)DepthFunction);

	glEnable(GL_CULL_FACE);
	glCullFace((unsigned int)CullMode);

	glPolygonMode((unsigned int)Graphics::CullMode::FrontBack, (unsigned int)RenderMode);

	if (MaterialShader && MaterialShader->IsValid()) {
		MaterialShader->Use();
	}
}
