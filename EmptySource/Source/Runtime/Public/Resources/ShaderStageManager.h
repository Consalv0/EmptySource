#pragma once

#include "Resources/ShaderStageResource.h"
#include "Resources/ResourceManager.h"

namespace EmptySource {

	class ShaderStageManager : public ResourceManager {
	private:
		typedef ResourceManager Supper;
		typedef TDictionary<size_t, ShaderStageResource *> ResourceDictionary;

		ShaderStageManager() : Supper(100u, RT_ShaderStage) {};

		ResourceDictionary Resources;

	public:

		void AddShaderStage(WString Name, EShaderType Type, WString FilePath);

		ShaderStageResource * GetResourceByUniqueName(const WString& Name) const { return GetResourceByUniqueID(WStringToHash(Name)); };

		ShaderStageResource * GetResourceByUniqueID(const size_t & UID) const;

		static ShaderStageManager & GetInstance();
	};

}