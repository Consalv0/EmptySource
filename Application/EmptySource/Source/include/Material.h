#pragma once

#include "../include/Graphics.h"
#include "../include/ShaderProgram.h"

class Material {
private:
	ShaderProgram* MaterialShader;

public:
	bool bUseDepthTest;
	Graphics::DepthFunction DepthFunction;
	Graphics::RenderMode RenderMode;
	Graphics::CullMode CullMode;

	Material();

	//* Set material shader
	void SetShaderProgram(ShaderProgram* Value);

	//* Get material shader
	ShaderProgram* GetShaderProgram() const;

	//* Pass Matrix4x4 Buffer Array
	void SetAttribMatrix4x4Array(const Char * AttributeName, int Count, const void* Data, const unsigned int& Buffer) const;

	//* Pass Matrix4x4 Array
	void SetMatrix4x4Array(const Char * UniformName, const float * Data, const int & Count = 1) const;

	//* Pass one float vector value array
	void SetFloat1Array(const Char * UniformName, const float * Data, const int & Count = 1) const;

	//* Pass two float vector value array
	void SetFloat2Array(const Char * UniformName, const float * Data, const int & Count = 1) const;

	//* Pass three float vector value array
	void SetFloat3Array(const Char * UniformName, const float * Data, const int & Count = 1) const;

	//* Pass four float vector value array
	void SetFloat4Array(const Char * UniformName, const float * Data, const int & Count = 1) const;

	//* Pass Texture 2D array
	void SetTexture2D(const Char * UniformName, struct Texture2D* Tex, const unsigned int& Position) const;

	//* Pass Cubemap array
	void SetTextureCubemap(const Char * UniformName, struct Cubemap* Tex, const unsigned int& Position) const;

	//* Use shader program and render mode
	void Use();
};
