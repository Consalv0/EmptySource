#pragma once

#include "Resources/ResourceManager.h"
#include "Resources/ShaderResource.h"
#include "Resources/ShaderParameters.h"

namespace EmptySource {

	class ShaderManager : public ResourceManager {
	public:
		using ShaderStageCode = std::pair<EShaderStageType, NString>;

		RShaderPtr GetProgram(const IName& Name) const;

		RShaderPtr GetProgram(const size_t & UID) const;

		void FreeShaderProgram(const IName& Name);

		void AddShaderProgram(RShaderPtr& Shader);

		TArray<IName> GetResourceShaderNames() const;

		virtual inline EResourceType GetResourceType() const override { return RT_Shader; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		RShaderPtr CreateProgram(const WString& Name, const WString & Origin, const NString& Source = "");

		static ShaderManager& GetInstance();

		static NString StageTypeToString(const EShaderStageType& Type);

		static EShaderStageType StringToStageType(const NString& Type);

	private:
		TArray<ShaderStageCode> GetStagesCodeFromSource(const NString& Source);

		TDictionary<size_t, IName> ShaderNameList;

		TDictionary<size_t, RShaderPtr> ShaderProgramList;
	};

}