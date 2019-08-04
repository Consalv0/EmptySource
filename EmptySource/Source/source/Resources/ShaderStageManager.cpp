#include "..\..\include\Resources\ShaderStageManager.h"

namespace EmptySource {

	void ShaderStageManager::AddShaderStage(WString Name, ShaderType Type, WString FilePath) {
		Resources.insert({ WStringToHash(Name), new ShaderStageResource(this, Name, Type, FilePath) });
	}

	ShaderStageResource * ShaderStageManager::GetResourceByUniqueID(const size_t & UID) const {
		auto Iterator = Resources.find(UID);
		if (Iterator == Resources.end())
			return NULL;
		else
			return Iterator->second;
	}

	ShaderStageManager & ShaderStageManager::GetInstance() {
		static ShaderStageManager Manager;
		return Manager;
	}

}