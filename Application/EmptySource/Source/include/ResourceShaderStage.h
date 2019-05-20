#pragma once

#ifdef RESOURCES_ADD_SHADERSTAGE

#include "../include/Resources.h"
#include "../include/ShaderStage.h"

struct ShaderStageData {
	size_t GUID;
	WString FilePath;
	ShaderType Type;
};

template<>
bool ResourceManager::ResourceToList<ShaderStageData>(const WString & File, ShaderStageData & ResourceData);

template<>
inline Resource<ShaderStage> * ResourceManager::Load(const WString & FilePath) {
	FileStream * File;
	if ((File = FileManager::GetFile(FilePath)) == NULL)
		return NULL;

	ShaderStageData LoadData;
	if (!ResourceToList<ShaderStageData>(FilePath, LoadData))
		return NULL;

	auto ResourceFind = Resources.find(GetHashName(FilePath));
	if (ResourceFind != Resources.end()) {
		return dynamic_cast<Resource<ShaderStage>*>(ResourceFind->second);
	}

	Resource<ShaderStage> * ResourceAdded = new Resource<ShaderStage>(FilePath, new ShaderStage(LoadData.Type, File));
	Resources.insert(std::pair<const size_t, BaseResource*>(ResourceAdded->GetIdentifierHash(), ResourceAdded));
	return ResourceAdded;
}
#endif // RESOURCES_ADD_SHADERSTAGE