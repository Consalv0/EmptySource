
#include "CoreMinimal.h"
#include "Resources/TextureManager.h"

#include <yaml-cpp/yaml.h>

namespace EmptySource {

	TexturePtr TextureManager::GetTexture(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetTexture(UID);
	}

	TexturePtr TextureManager::GetTexture(const size_t & UID) const {
		auto Resource = TextureList.find(UID);
		if (Resource != TextureList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void TextureManager::FreeTexture(const WString & Name) {
		size_t UID = WStringToHash(Name);
		TextureList.erase(UID);
	}

	void TextureManager::AddTexture(const WString& Name, TexturePtr Tex) {
		size_t UID = WStringToHash(Name);
		TextureList.insert({ UID, Tex });
	}

	void TextureManager::LoadResourcesFromFile(const WString & FilePath) {
		FileStream * ResourcesFile = ResourceManager::GetResourcesFile(FilePath);

		YAML::Node BaseNode;
		{
			NString FileInfo;
			if (ResourcesFile == NULL || !ResourcesFile->ReadNarrowStream(&FileInfo))
				return;
			BaseNode = YAML::Load(FileInfo.c_str());
		}

		YAML::Node ResourcesNode = BaseNode["Resources"];

		if (ResourcesNode.IsDefined()) {
			int FileNodePos = -1;

			for (size_t i = 0; i < ResourcesNode.size(); i++) {
				if (ResourcesNode[i]["GUID"].IsDefined() && ResourcesNode[i]["GUID"]["ShaderStage"].IsDefined()) {
					FileNodePos = (int)i;

					YAML::Node ShaderStageNode = ResourcesNode[FileNodePos]["ShaderStage"];
					WString FilePath = ShaderStageNode["FilePath"].IsDefined() ? Text::NarrowToWide(ShaderStageNode["FilePath"].as<NString>()) : L"";
					NString Type = ShaderStageNode["Type"].IsDefined() ? ShaderStageNode["Type"].as<NString>() : "Vertex";

					// AddShaderStage(FilePath, ShaderStage::CreateFromFile(FilePath, ShaderType));
				}
			}

			if (FileNodePos < 0)
				return;

			for (size_t i = 0; i < ResourcesNode.size(); i++) {
				if (ResourcesNode[i]["GUID"].IsDefined() && ResourcesNode[i]["GUID"]["ShaderProgram"].IsDefined()) {
					FileNodePos = (int)i;

					YAML::Node ShaderProgramNode = ResourcesNode[FileNodePos]["ShaderProgram"];
					WString Name = ShaderProgramNode["Name"].IsDefined() ? Text::NarrowToWide(ShaderProgramNode["Name"].as<NString>()) : L"";
					// AddShaderStage(FilePath, ShaderStage::CreateFromFile(FilePath, ShaderType));
				}
			}
		}
		else return;
	}

	TextureManager & TextureManager::GetInstance() {
		static TextureManager Manager;
		return Manager;
	}

}
