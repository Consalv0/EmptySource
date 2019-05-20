
#include "../include/FileStream.h"
#include "../include/Resources.h"
#include "../include/FileManager.h"
#include "../include/ShaderStage.h"

#include "../External/YAML/include/yaml-cpp/yaml.h"

TDictionary<size_t, BaseResource*> ResourceManager::Resources = TDictionary<size_t, BaseResource*>();

BaseResource::BaseResource(const WString & FilePath) : IIdentifier(FilePath), FilePath(FilePath), isDone(false) {
}

bool BaseResource::IsDone() {
	return isDone;
}

const FileStream * BaseResource::GetFile() {
	return FileManager::GetFile(FilePath);
}