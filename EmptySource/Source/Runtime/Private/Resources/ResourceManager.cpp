
#include "CoreMinimal.h"
#include "Resources/ResourceManager.h"

#include <yaml-cpp/yaml.h>

namespace EmptySource {
	
	FileStream * ResourceManager::GetResourcesFile(const WString & ResourceFilePath) {
		FileStream * ResourceFile = FileManager::GetFile(ResourceFilePath);

		YAML::Node BaseNode;
		if (ResourceFile == NULL) {
			ResourceFile = FileManager::MakeFile(ResourceFilePath);
		}
		else {
			NString FileInfo;
			if (!ResourceFile->ReadNarrowStream(&FileInfo))
				return NULL;
			try {
				BaseNode = YAML::Load(FileInfo.c_str());
				if (!BaseNode["Resources"].IsDefined()) {
					return NULL;
				}
			}
			catch (...) {
				LOG_CORE_WARN(L"The {} file could not be parsed", ResourceFilePath);
				return NULL;
			}
		}

		return ResourceFile;
	}

	FileStream * ResourceManager::CreateResourcesFile(const WString & ResourceFilePath) {
		FileStream * ResourceFile = FileManager::GetFile(ResourceFilePath);

		YAML::Node BaseNode;
		if (ResourceFile == NULL) {
			ResourceFile = FileManager::MakeFile(ResourceFilePath);
		}

		if (ResourceFile->IsValid()) {
			ResourceFile->Clean();
			YAML::Node ResourcesNode;
			BaseNode["Resources"] = ResourcesNode;
			YAML::Emitter Out;
			Out << BaseNode;
			(*ResourceFile) << Out.c_str();
			ResourceFile->Close();
		}

		return ResourceFile;
	}

}