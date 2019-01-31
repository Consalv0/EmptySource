#pragma once

#include "..\include\Graphics.h"
#include "..\include\ShaderProgram.h"

class Material {
private:
	ShaderProgram* MaterialShader;

public:
	bool bUseDepthTest;
	Render::DepthFunction DepthFunction;
	Render::RenderMode RenderMode;
	Render::CullMode CullMode;

	Material();

	/*Set material shader*/
	void SetShaderProgram(ShaderProgram* Value);

	/*Use shader program and render mode*/
	void Use();
};