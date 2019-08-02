
#include "../include/ResourceManager.h"
#include "../include/Utility/LogCore.h"

#include "../External/YAML/include/yaml-cpp/yaml.h"

FileStream * ResourceManager::GetResourcesFile(const WString & ResourceFilePath) {
	FileStream * ResourceFile = FileManager::GetFile(ResourceFilePath);

	YAML::Node BaseNode;
	if (ResourceFile == NULL) {
		ResourceFile = FileManager::MakeFile(ResourceFilePath);
	} else {
		String FileInfo;
		if (!ResourceFile->ReadNarrowStream(&FileInfo))
			return NULL;
		try {
			BaseNode = YAML::Load(FileInfo.c_str());
			if (!BaseNode["Resources"].IsDefined()) {
				return NULL;
			}
		}
		catch (...) {
			Debug::Log(Debug::LogWarning, L"The %ls file could not be parsed", ResourceFilePath.c_str());
			return NULL;
		}
	}

	return ResourceFile;
}

FileStream * ResourceManager::CreateResourceFile(const WString & ResourceFilePath) {
	FileStream * ResourceFile = FileManager::GetFile(ResourceFilePath);

	YAML::Node BaseNode;
	if (ResourceFile == NULL) {
		ResourceFile = FileManager::MakeFile(ResourceFilePath);
	}

	if (ResourceFile->IsValid()) {
		ResourceFile->Clean();
		YAML::Node GroupsNode;
		BaseNode["Groups"] = GroupsNode;
		YAML::Node ResourcesNode;
		BaseNode["Resources"] = ResourcesNode;
		YAML::Emitter Out;
		Out << BaseNode;
		(*ResourceFile) << Out.c_str();
		ResourceFile->Close();
	}

	return ResourceFile;
}
