#pragma once

#include "RenderingDefinitions.h"
#include "Rendering/RenderingBuffers.h"
#include "Rendering/Texture.h"

namespace ESource {

	class ShaderStage {
	public:
		virtual ~ShaderStage() = default;

		//* Get the shader type
		virtual inline EShaderStageType GetType() const = 0;

		//* Get the shader object
		virtual void * GetStageObject() const = 0;

		//* The stage is valid
		virtual bool IsValid() const = 0;

		//* Create and compile our shader unit
		static ShaderStage * CreateFromText(const NString & Code, EShaderStageType Type);
	};

	class ShaderProgram {
	public:
		virtual void Bind() const = 0;

		virtual void Unbind() const = 0;

		virtual ~ShaderProgram() = default;

		//* Get the location id of a uniform variable in this shader
		virtual uint32_t GetUniformLocation(const NChar* LocationName) = 0;

		//* Get the location of the attrib in this shader
		virtual uint32_t GetAttribLocation(const NChar* LocationName) = 0;

		//* Pass Matrix4x4 Buffer Array
		virtual void SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void* Data, const VertexBufferPtr& Buffer) = 0;

		//* Pass Matrix4x4 Array
		virtual void SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass one float vector value array
		virtual void SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass one int vector value array
		virtual void SetInt1Array(const NChar * UniformName, const int * Data, const int & Count = 1) = 0;

		//* Pass two float vector value array
		virtual void SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass three float vector value array
		virtual void SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass four float vector value array
		virtual void SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count = 1) = 0;

		//* Pass Texture
		virtual void SetTexture(const NChar * UniformName, Texture * Tex, const uint32_t& Position) = 0;
		
		//* Get the shader object
		virtual void * GetShaderObject() const = 0;

		//* The shader is valid for use?
		virtual inline bool IsValid() const = 0;

		static ShaderProgram * Create(TArray<ShaderStage *> ShaderStages);
	};

}