#pragma once

#include "Resources/ResourceManager.h"
#include "ResourceShader.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	class ShaderManager : public ResourceManager {

	public:
		ShaderPtr GetProgram(const WString& Name) const;

		ShaderPtr GetProgram(const size_t & UID) const;

		ShaderStagePtr GetStage(const WString& Name) const;

		ShaderStagePtr GetStage(const size_t & UID) const;

		void FreeShaderProgram(const WString& Name);

		void FreeShaderStage(const WString& Name);

		void AddShaderProgram(ShaderPtr& Shader);

		void AddShaderStage(const WString & Name, ShaderStagePtr& Stage);

		virtual inline EResourceType GetResourceType() const override { return RT_Shader; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		static ShaderManager& GetInstance();

	private:
		TDictionary<size_t, ShaderPtr> ShaderProgramList;

		TDictionary<size_t, ShaderStagePtr> ShaderStageList;
	};

}