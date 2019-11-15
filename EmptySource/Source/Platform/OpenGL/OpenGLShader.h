#pragma once

namespace ESource {

	class OpenGLShaderStage : public ShaderStage {
	public:

		OpenGLShaderStage(const NString & Code, EShaderStageType Type, int LineCountOffset, int Flags);

		virtual ~OpenGLShaderStage() override;

		//* Get the shader type
		virtual inline EShaderStageType GetType() const override { return StageType; };

		//* Get the shader object
		virtual void * GetStageObject() const override { return (void *)(unsigned long long)ShaderObject; };

		//* The shader is valid for use?
		virtual inline bool IsValid() const override { return bValid; };
	
	private:
		EShaderStageType StageType;

		uint32_t ShaderObject;

		bool bValid;

		void PreprocessShaderStage(NString & Code, EShaderStageType Type, int LineCountOffset, bool Instancing);
	};

	class OpenGLShaderProgram : public ShaderProgram {
	public:
		OpenGLShaderProgram(TArray<ShaderStage *> ShaderStages);

		~OpenGLShaderProgram();

		virtual void Bind() const override;

		virtual void Unbind() const override;

		//* Get the location id of a uniform variable in this shader
		virtual uint32_t GetUniformLocation(const NChar* LocationName) override;

		//* Get the location of the attrib in this shader
		virtual uint32_t GetAttribLocation(const NChar* LocationName) override;

		//* Pass Matrix4x4 Buffer Array
		virtual void SetAttribMatrix4x4Array(const NChar * AttributeName, int Count, const void* Data, const VertexBufferPtr& Buffer) override;

		//* Pass Matrix4x4 Array
		virtual void SetMatrix4x4Array(const NChar * UniformName, const float * Data, const int & Count = 1) override;

		//* Pass one float vector value array
		virtual void SetFloat1Array(const NChar * UniformName, const float * Data, const int & Count = 1) override;

		//* Pass one int vector value array
		virtual void SetInt1Array(const NChar * UniformName, const int * Data, const int & Count = 1) override;

		//* Pass two float vector value array
		virtual void SetFloat2Array(const NChar * UniformName, const float * Data, const int & Count = 1) override;

		//* Pass three float vector value array
		virtual void SetFloat3Array(const NChar * UniformName, const float * Data, const int & Count = 1) override;

		//* Pass four float vector value array
		virtual void SetFloat4Array(const NChar * UniformName, const float * Data, const int & Count = 1) override;

		//* Pass Texture 2D array
		virtual void SetTexture(const NChar * UniformName, Texture * Tex, const uint32_t& Position) override;

		//* Get the shader object
		virtual void * GetShaderObject() const override { return (void *)(unsigned long long)ProgramObject; }

		//* The shader is valid for use?
		virtual inline bool IsValid() const override { return bValid; };

	private:
		typedef TDictionary<NString, uint32_t> LocationMap;
		
		//* Shader Program Object
		uint32_t ProgramObject;

		bool bValid;

		LocationMap Locations;

	};

}