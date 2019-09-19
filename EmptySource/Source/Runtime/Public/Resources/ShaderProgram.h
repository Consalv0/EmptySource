#pragma once

#include "Resources/ResourceManager.h"
#include "ResourceHolder.h"
#include "Resources/ShaderParameters.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	typedef std::shared_ptr<class RShaderStage> RShaderStagePtr;

	struct ResourceShaderStageData {
		WString Name;
		EShaderStageType ShaderType;
		NString Code;
		WString FilePath;
		WString Origin;

		ResourceShaderStageData() = default;

		ResourceShaderStageData(const WString& FilePath, size_t UID);
	};

	class RShaderStage : public ResourceHolder {
	public:
		~RShaderStage();

		virtual bool IsValid() override;

		virtual void Load() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }

		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Shader; }

		virtual inline size_t GetMemorySize() const override { return sizeof(unsigned int); };

		inline ShaderStage * GetShaderStage() { return StagePointer; };

		static inline EResourceType GetType() { return EResourceType::RT_Shader; };

		inline EShaderStageType & GetShaderType() { return StageType; };

	protected:
		friend class ShaderManager;

		RShaderStage(const IName & Name, const WString & Origin, EShaderStageType Type, const NString & Code);

	private:
		ShaderStage * StagePointer;
		EShaderStageType StageType;
		NString SourceCode;
	};

	struct ResourceShaderData {
	public:
		WString Name;
		RShaderStagePtr VertexShader;
		RShaderStagePtr PixelShader;
		RShaderStagePtr ComputeShader;
		RShaderStagePtr GeometryShader;
		WString Origin;

		ResourceShaderData() = default;

		ResourceShaderData(const WString& FilePath, size_t UID);
	};

	typedef std::shared_ptr<class RShaderProgram> RShaderProgramPtr;

	class RShaderProgram : public ResourceHolder {
	public:
		~RShaderProgram();

		virtual bool IsValid() override;

		virtual void Load() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }
		
		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Shader; }

		virtual inline size_t GetMemorySize() const override { return sizeof(unsigned int); };

		inline ShaderProgram * GetProgram() { return ShaderPointer; };

		static inline EResourceType GetType() { return EResourceType::RT_Shader; };

		//* Get the chader properties
		const TArray<ShaderProperty> & GetProperties() const { return Properties; }

		//* Set the shader properties
		void SetProperties(const TArrayInitializer<ShaderProperty> & Properties);

		//* Set the shader properties
		void SetProperties(const TArray<ShaderProperty> & Properties);

	protected:
		friend class ShaderManager;

		RShaderProgram(const IName & Name, const WString & Origin, TArray<RShaderStagePtr>& Stages);

	private:
		ShaderProgram * ShaderPointer;
		RShaderStagePtr VertexShader;
		RShaderStagePtr PixelShader;
		RShaderStagePtr ComputeShader;
		RShaderStagePtr GeometryShader;

		TArray<ShaderProperty> Properties;
	};

}
