
#include "../include/FileStream.h"
#include "../include/Resources.h"
#include "../include/FileManager.h"
#include "../include/ShaderStage.h"

#include "../External/YAML/include/yaml-cpp/yaml.h"

TDictionary<size_t, BaseResource*> ResourceManager::Resources = TDictionary<size_t, BaseResource*>();

BaseResource::BaseResource(const WString & Name, const size_t & GUID) : GUID(GUID), Name(Name), isDone(false) {
}

bool BaseResource::IsDone() const {
	return isDone;
}

size_t BaseResource::GetIdentifier() const {
	return GUID;
}

const FileStream * BaseResource::GetFile() const {
	return FileManager::GetFile(Name);
}

FileStream * ResourceManager::GetResourcesFile() {
	static WString ResourceFilePath = L"Resources/Resouces.yaml";
	FileStream * ResourceFile = FileManager::GetFile(ResourceFilePath);
	YAML::Node BaseNode;
	bool bNeedsModification = false;
	if (ResourceFile == NULL) {
		ResourceFile = FileManager::MakeFile(ResourceFilePath);
		bNeedsModification = true;
	}
	else {
		String FileInfo;
		if (!ResourceFile->ReadNarrowStream(&FileInfo))
			return NULL;
		BaseNode = YAML::Load(FileInfo.c_str());
		if (!BaseNode["Resources"].IsDefined()) {
			bNeedsModification = true;
		}
	}

	if (bNeedsModification && ResourceFile->IsValid()) {
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
