#pragma once

#include "..\include\RenderingCore.h"
#include "..\include\Shader.h"

class Material {
private:
	Shader* MaterialShader;

public:
	bool bUseDepthTest;
	Render::DepthFunction DepthFunction;
	Render::RenderMode RenderMode;
	Render::CullMode CullMode;

	Material();

	/*Set material shader*/
	void SetShader(Shader* Value);

	/*Use shader program and render mode*/
	void Use();
};