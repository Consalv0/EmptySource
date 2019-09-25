
#include "CoreMinimal.h"
#include "Resources/TextureManager.h"
#include "Resources/ImageConversion.h"

#include <yaml-cpp/yaml.h>

namespace EmptySource {

	RTexturePtr TextureManager::GetTexture(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetTexture(UID);
	}

	RTexturePtr TextureManager::GetTexture(const size_t & UID) const {
		auto Resource = TextureList.find(UID);
		if (Resource != TextureList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void TextureManager::FreeTexture(const WString & Name) {
		size_t UID = WStringToHash(Name);
		TextureNameList.erase(UID);
		TextureList.erase(UID);
	}

	void TextureManager::AddTexture(const WString& Name, RTexturePtr Tex) {
		size_t UID = WStringToHash(Name);
		TextureNameList.insert({ UID, Name });
		TextureList.insert({ UID, Tex });
	}

	TArray<WString> TextureManager::GetResourceNames() const {
		TArray<WString> Names;
		for (auto KeyValue : TextureNameList)
			Names.push_back(KeyValue.second);
		std::sort(Names.begin(), Names.end(), [](const WString& first, const WString& second) {
			unsigned int i = 0;
			while ((i < first.length()) && (i < second.length())) {
				if (tolower(first[i]) < tolower(second[i])) return true;
				else if (tolower(first[i]) > tolower(second[i])) return false;
				++i;
			}
			return (first.length() < second.length());
		});
		return Names;
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

	void TextureManager::LoadImageFromFile(
		const WString& Name, EColorFormat ColorFormat, EFilterMode FilterMode,
		ESamplerAddressMode AddressMode, bool bFlipVertically, bool bGenMipMaps, const WString & FilePath) {

		// LOG_CORE_DEBUG(L"Loading Texture2D {}...", FileManager::GetFile(FilePath)->GetFileName().c_str());
		// EColorFormat InColorFormat = ImageConversion::GetColorFormat(FileManager::GetFile(FilePath));
		// RTexturePtr LoadedTexture = NULL;
		// PixelMap Bitmap;
		// ImageConversion::LoadFromFile(Bitmap, FileManager::GetFile(FilePath), InColorFormat, bFlipVertically);
		// LoadedTexture = Texture2D::Create(Name, Bitmap.GetSize(), ColorFormat, FilterMode, AddressMode, InColorFormat, Bitmap.PointerToValue());
		// 
		// if (LoadedTexture) {
		// 	if (bGenMipMaps) LoadedTexture->GenerateMipMaps();
		// 	AddTexture(Name, LoadedTexture);
		// }
	}

	TextureManager & TextureManager::GetInstance() {
		static TextureManager Manager;
		return Manager;
	}

}
