
#include "Files/FileStream.h"
#include "Files/FileManager.h"
#include "Resources/Resources.h"
#include "Graphics/ShaderStage.h"

#include "../External/YAML/include/yaml-cpp/yaml.h"

namespace EmptySource {

	TDictionary<size_t, BaseResource*> OldResourceManager::Resources = TDictionary<size_t, BaseResource*>();

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

	FileStream * OldResourceManager::GetResourcesFile() {
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
			try {
				BaseNode = YAML::Load(FileInfo.c_str());
				if (!BaseNode["Resources"].IsDefined()) {
					bNeedsModification = true;
				}
			}
			catch (...) {
				Debug::Log(Debug::LogWarning, L"The Resource.yaml file could not be parsed");
				FileStream * ResourceFileSave = FileManager::MakeFile(ResourceFilePath + L".save");
				ResourceFileSave->Clean();
				(*ResourceFileSave) << FileInfo.c_str();
				ResourceFileSave->Close();
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

}