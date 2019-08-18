#pragma once

#include "Rendering/RenderingDefinitions.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	class Material {
	private:
		ShaderPtr MaterialShader;

	public:
		bool bUseDepthTest;
		unsigned int RenderPriority;
		EDepthFunction DepthFunction;
		ERasterizerFillMode FillMode;
		ECullMode CullMode;

		Material();

		//* Set material shader
		void SetShaderProgram(ShaderPtr Value);

		//* Get material shader
		ShaderPtr GetShaderProgram() const;

		//* Pass Matrix4x4 Buffer Array
		void SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void* Data, const unsigned int& Buffer) const;

		//* Pass Matrix4x4 Array
		void SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass one float vector value array
		void SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass one int vector value array
		void SetInt1Array(const NChar * UniformName, const int * Data, const int & Count = 1) const;

		//* Pass two float vector value array
		void SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass three float vector value array
		void SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass four float vector value array
		void SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count = 1) const;

		//* Pass Texture 2D array
		void SetTexture2D(const NChar * UniformName, struct Texture2D* Tex, const unsigned int& Position) const;

		//* Pass Cubemap array
		void SetTextureCubemap(const NChar * UniformName, struct Cubemap* Tex, const unsigned int& Position) const;

		//* Use shader program and render mode
		void Use();
	};

}