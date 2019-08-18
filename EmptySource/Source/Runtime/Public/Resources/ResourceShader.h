#pragma once

#include "Resources/ResourceManager.h"
#include "ResourceHolder.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	typedef std::shared_ptr<class ResourceHolderShaderStage> ResourceHolderShaderStagePtr;

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

	class ResourceHolderShaderStage : public ResourceHolder {
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

		ResourceHolderShaderStage(const ResourceShaderStageData & Data);

		ResourceHolderShaderStage(ShaderStagePtr & Data);

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

	typedef std::shared_ptr<class ResourceHolderShader> ResourceHolderShaderPtr;

	class ResourceHolderShader : public ResourceHolder {
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

		ResourceHolderShader(const ResourceShaderData& Data);

		ResourceHolderShader(const WString & Name, TArray<ShaderStagePtr> Stages);

		ResourceHolderShader(ShaderPtr & Data);

	private:
		ShaderPtr Shader;
		EResourceLoadState LoadState;
		ResourceShaderData Data;
	};

}
