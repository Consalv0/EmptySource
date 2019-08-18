#pragma once

#include "Resources/ResourceManager.h"
#include "ResourceHolder.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	typedef std::shared_ptr<class ShaderStageResource> ResourceHolderShaderStagePtr;

	struct ResourceShaderStageData {
	public:
		WString Name;
		EShaderType ShaderType;
		NString Code;
		WString FilePath;
		WString ResourceFile;

		ResourceShaderStageData() = default;

		ResourceShaderStageData(const WString& FilePath, size_t UID);
	};

	class ShaderStageResource : public Resource {
	public:
		virtual void Load() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }

		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Shader; }

		inline ShaderStagePtr GetShaderStage() { return Stage; };

		static inline EResourceType GetType() { return EResourceType::RT_Shader; };

	protected:
		friend class ResourceManager;

		ShaderStageResource(const ResourceShaderStageData & Data);

		ShaderStageResource(ShaderStagePtr & Data);

	private:
		ShaderStagePtr Stage;
		EResourceLoadState LoadState;
		ResourceShaderStageData Data;
	};

	struct ResourceShaderData {
	public:
		WString Name;
		ShaderStagePtr VertexShader;
		ShaderStagePtr PixelShader;
		ShaderStagePtr ComputeShader;
		ShaderStagePtr GeometryShader;
		WString ResourceFile;

		ResourceShaderData() = default;

		ResourceShaderData(const WString& FilePath, size_t UID);
	};

	typedef std::shared_ptr<class ShaderProgramResource> ResourceHolderShaderPtr;

	class ShaderProgramResource : public Resource {
	public:
		virtual void Load() override;

		virtual void Unload() override;

		virtual void Reload() override;

		virtual inline EResourceLoadState GetLoadState() const override { return LoadState; }
		
		virtual inline EResourceType GetResourceType() const override { return EResourceType::RT_Shader; }

		inline ShaderPtr GetShader() { return Shader; };

		static inline EResourceType GetType() { return EResourceType::RT_Shader; };

	protected:
		friend class ResourceManager;

		ShaderProgramResource(const ResourceShaderData& Data);

		ShaderProgramResource(const WString & Name, TArray<ShaderStagePtr> Stages);

		ShaderProgramResource(ShaderPtr & Data);

	private:
		ShaderPtr Shader;
		EResourceLoadState LoadState;
		ResourceShaderData Data;
	};

}
