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

		virtual inline WString GetName() const override { return Name; }

		//* Get the shader object
		virtual void * GetShaderObject() const override { return (void *)(unsigned long long)ProgramObject; }

		//* The shader is valid for use?
		virtual inline bool IsValid() const override { return bValid; };

	protected:
		//* Appends shader unit to shader program
		virtual void AppendStage(ShaderStagePtr ShaderProgram) override;

	private:
		typedef TDictionary<const NChar *, unsigned int> LocationMap;
		
		ShaderStagePtr VertexShader;
		ShaderStagePtr PixelShader;
		ShaderStagePtr ComputeShader;
		ShaderStagePtr GeometryShader;
		
		WString Name;
		
		//* Shader Program Object
		unsigned int ProgramObject;

		bool bValid;

		LocationMap Locations;

	};

}