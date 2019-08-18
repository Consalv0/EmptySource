#pragma once

#include "Resources/ResourceManager.h"
#include "ResourceShader.h"
#include "Rendering/Shader.h"

namespace EmptySource {

	class ShaderManager : public ResourceManager {

	public:
		ShaderPtr GetProgramByName(const WString& Name) const;

		ShaderPtr GetProgramByUniqueID(const size_t & UID) const;

		ShaderStagePtr GetStageByName(const WString& Name) const;

		ShaderStagePtr GetStageByUniqueID(const size_t & UID) const;

		void AddShaderProgram(ShaderPtr& Shader);

		void AddShaderStage(const WString & Name, ShaderStagePtr& Stage);

		virtual inline EResourceType GetResourceType() const override { return RT_Shader; };

		virtual void GetResourcesFromFile(const WString& FilePath) override;

		static ShaderManager& GetInstance();

	private:
		TDictionary<size_t, ShaderPtr> ShaderProgramList;
		TDictionary<size_t, ShaderStagePtr> ShaderStageList;

	};

}