#pragma once

#include "Resources/ResourceManager.h"
#include "Resources/ShaderProgram.h"
#include "Resources/ShaderParameters.h"

namespace EmptySource {

	class ShaderManager : public ResourceManager {

	public:
		RShaderProgramPtr GetProgram(const IName& Name) const;

		RShaderProgramPtr GetProgram(const size_t & UID) const;

		RShaderStagePtr GetStage(const IName& Name) const;

		RShaderStagePtr GetStage(const size_t & UID) const;

		void FreeShaderProgram(const IName& Name);

		void FreeShaderStage(const IName& Name);

		void AddShaderProgram(RShaderProgramPtr& Shader);

		void AddShaderStage(const IName & Name, RShaderStagePtr& Stage);

		TArray<IName> GetResourceShaderNames() const;

		TArray<IName> GetResourceShaderStageNames() const;

		virtual inline EResourceType GetResourceType() const override { return RT_Shader; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		RShaderStagePtr LoadStage(const WString & Name, const WString & Origin, EShaderStageType Type);

		RShaderProgramPtr LoadProgram(const WString & Name, const WString & Origin, TArray<RShaderStagePtr> & Stages);

		static ShaderManager& GetInstance();

	private:
		TDictionary<size_t, IName> ShaderNameList;
		TDictionary<size_t, IName> ShaderStageNameList;

		TDictionary<size_t, RShaderProgramPtr> ShaderProgramList;
		TDictionary<size_t, RShaderStagePtr> ShaderStageList;
	};

}