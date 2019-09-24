#pragma once

#include "Resources/ResourceManager.h"
#include "Resources/ShaderProgram.h"
#include "Resources/ShaderParameters.h"

namespace EmptySource {

	class ShaderManager : public ResourceManager {
	public:
		using ShaderStageCode = std::pair<EShaderStageType, NString>;

		RShaderProgramPtr GetProgram(const IName& Name) const;

		RShaderProgramPtr GetProgram(const size_t & UID) const;

		void FreeShaderProgram(const IName& Name);

		void AddShaderProgram(RShaderProgramPtr& Shader);

		TArray<IName> GetResourceShaderNames() const;

		virtual inline EResourceType GetResourceType() const override { return RT_Shader; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		RShaderProgramPtr CreateProgram(const WString& Name, const WString & Origin, const NString& Source = "");

		static ShaderManager& GetInstance();

		static NString StageTypeToString(const EShaderStageType& Type);

		static EShaderStageType StringToStageType(const NString& Type);

	private:
		TArray<ShaderStageCode> GetStagesCodeFromSource(const NString& Source);

		TDictionary<size_t, IName> ShaderNameList;

		TDictionary<size_t, RShaderProgramPtr> ShaderProgramList;
	};

}