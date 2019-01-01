#include "..\include\Core.h"
#include "..\include\Material.h"

Material::Material() {
	MaterialShader = NULL;
	bUseDepthTest = true;
	DepthFunction = Render::DepthFunction::LessEqual;
	RenderMode = Render::RenderMode::Fill;
	CullMode = Render::CullMode::FrontBack;
}

void Material::SetShader(Shader* Value) {
	MaterialShader = Value;
}

void Material::Use() {
	// Activate Z-buffer
	if (bUseDepthTest) {
		glEnable(GL_DEPTH_TEST);
	} else {
		glDisable(GL_DEPTH_TEST);
	}

	glDepthFunc((unsigned int)DepthFunction);

	glPolygonMode((unsigned int)CullMode, (unsigned int)RenderMode);

	if (MaterialShader && MaterialShader->IsValid()) {
		MaterialShader->Use();
	}
}
