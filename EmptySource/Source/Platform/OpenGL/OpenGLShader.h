#pragma once

namespace EmptySource {

	class OpenGLShaderStage : public ShaderStage {
	public:
		OpenGLShaderStage(const WString & FilePath, EShaderType Type);

		OpenGLShaderStage(const NString & Code, EShaderType Type);

		virtual ~OpenGLShaderStage() override;

		//* Get the shader type
		virtual inline EShaderType GetType() const override { return StageType; };

		//* Get the shader object
		virtual void * GetStageObject() const override { return (void *)(unsigned long long)ShaderObject; };

		//* The shader is valid for use?
		virtual inline bool IsValid() const override { return bValid; };
	
	private:
		EShaderType StageType;

		unsigned int ShaderObject;

		bool bValid;
	};

	class OpenGLShaderProgram : public ShaderProgram {
	public:
		OpenGLShaderProgram(const WString& Name, TArray<ShaderStagePtr> ShaderStages);

		~OpenGLShaderProgram();

		virtual void Bind() const override;

		virtual void Unbind() const override;

		//* Get the location id of a uniform variable in this shader
		virtual unsigned int GetUniformLocation(const NChar* LocationName) override;

		//* Get the location of the attrib in this shader
		virtual unsigned int GetAttribLocation(const NChar* LocationName) override;

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
		virtual void SetTexture2D(const NChar * UniformName, TexturePtr Tex, const unsigned int& Position) override;

		//* Pass Cubemap array
		virtual void SetTextureCubemap(const NChar * UniformName, TexturePtr Tex, const unsigned int& Position) override;

		virtual inline WString GetName() const override { return Name; }

		//* Get the chader properties
		virtual const TArray<ShaderProperty> & GetProperties() const override { return Properties; }

		//* Set the shader properties
		virtual void SetProperties(const TArrayInitializer<ShaderProperty> & Properties) override;
		//* Set the shader properties
		virtual void SetProperties(const TArray<ShaderProperty> & Properties) override;

		//* Get the shader object
		virtual void * GetShaderObject() const override { return (void *)(unsigned long long)ProgramObject; }

		//* The shader is valid for use?
		virtual inline bool IsValid() const override { return bValid; };

	protected:
		//* Appends shader unit to shader program
		virtual void AppendStage(ShaderStagePtr ShaderProgram) override;

	private:
		typedef TDictionary<NString, unsigned int> LocationMap;
		
		ShaderStagePtr VertexShader;
		ShaderStagePtr PixelShader;
		ShaderStagePtr ComputeShader;
		ShaderStagePtr GeometryShader;

		TArray<ShaderProperty> Properties;
		
		WString Name;
		
		//* Shader Program Object
		unsigned int ProgramObject;

		bool bValid;

		LocationMap Locations;

	};

}