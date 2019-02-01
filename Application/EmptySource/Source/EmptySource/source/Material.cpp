#include "..\include\Core.h"
#include "..\include\Material.h"

Material::Material() {
	MaterialShader = NULL;
	bUseDepthTest = true;
	DepthFunction = Render::DepthFunction::LessEqual;
	RenderMode = Render::RenderMode::Fill;
	CullMode = Render::CullMode::Back;
}

void Material::SetShaderProgram(ShaderProgram* Value) {
	if (Value && Value->IsValid()) {
		MaterialShader = Value;
	} else {
		Debug::Log(Debug::LogError, L"The ShaderProgram is NULL or is not valid");
	}
}

ShaderProgram * Material::GetShaderProgram() const {
	return MaterialShader;
}

void Material::Use() {
	// Activate Z-buffer
	if (bUseDepthTest) {
		glEnable(GL_DEPTH_TEST);
	} else {
		glDisable(GL_DEPTH_TEST);
	}

	glDepthFunc((unsigned int)DepthFunction);

	glEnable(GL_CULL_FACE);
	glCullFace((unsigned int)CullMode);

	glPolygonMode((unsigned int)Render::CullMode::FrontBack, (unsigned int)RenderMode);

	if (MaterialShader && MaterialShader->IsValid()) {
		MaterialShader->Use();
	}
}
