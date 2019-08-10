#pragma once

#ifdef RESOURCES_ADD_SHADERSTAGE

#include "Resources/Resources.h"
#include "Rendering/ShaderStage.h"

namespace EmptySource {

	struct ShaderStageData {
		size_t GUID;
		WString FilePath;
		EShaderType Type;
	};

	template<>
	bool OldResourceManager::GetResourceData<ShaderStageData>(const WString & File, ShaderStageData & ResourceData);

	template<>
	inline Resource<ShaderStage> * OldResourceManager::Load(const WString & FilePath) {
		FileStream * File;
		if ((File = FileManager::GetFile(FilePath)) == NULL)
			return NULL;

		ShaderStageData LoadData;
		if (!GetResourceData<ShaderStageData>(FilePath, LoadData))
			return NULL;

		auto ResourceFind = Resources.find(LoadData.GUID);
		if (ResourceFind != Resources.end()) {
			return dynamic_cast<Resource<ShaderStage>*>(ResourceFind->second);
		}

		Resource<ShaderStage> * ResourceAdded = new Resource<ShaderStage>(LoadData.FilePath, LoadData.GUID, new ShaderStage(LoadData.Type));
		ResourceAdded->GetData()->CompileFromFile(FilePath);
		Resources.insert(std::pair<const size_t, BaseResource*>(ResourceAdded->GetIdentifier(), ResourceAdded));
		return ResourceAdded;
	}

	template<>
	bool OldResourceManager::GetResourceData<ShaderStageData>(const size_t & GUID, ShaderStageData & ResourceData);

	template<>
	inline Resource<ShaderStage> * OldResourceManager::Load(const size_t & GUID) {
		ShaderStageData LoadData;
		if (!GetResourceData<ShaderStageData>(GUID, LoadData))
			return NULL;

		FileStream * File;
		if ((File = FileManager::GetFile(LoadData.FilePath)) == NULL)
			return NULL;

		auto ResourceFind = Resources.find(LoadData.GUID);
		if (ResourceFind != Resources.end()) {
			return dynamic_cast<Resource<ShaderStage>*>(ResourceFind->second);
		}

		Resource<ShaderStage> * ResourceAdded = new Resource<ShaderStage>(LoadData.FilePath, LoadData.GUID, new ShaderStage(LoadData.Type));
		ResourceAdded->GetData()->CompileFromFile(LoadData.FilePath);
		Resources.insert(std::pair<const size_t, BaseResource*>(ResourceAdded->GetIdentifier(), ResourceAdded));
		return ResourceAdded;
	}

}
#endif // RESOURCES_ADD_SHADERSTAGE