
#include "CoreMinimal.h"
#include "Resources/TextureManager.h"
#include "Resources/ImageConversion.h"

#include <yaml-cpp/yaml.h>

namespace ESource {

	RTexturePtr TextureManager::GetTexture(const IName & Name) const {
		return GetTexture(Name.GetID());
	}

	RTexturePtr TextureManager::GetTexture(const size_t & UID) const {
		auto Resource = TextureList.find(UID);
		if (Resource != TextureList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void TextureManager::FreeTexture(const IName & Name) {
		size_t ID = Name.GetID();
		TextureNameList.erase(ID);
		TextureList.erase(ID);
	}

	void TextureManager::AddTexture(RTexturePtr Tex) {
		size_t ID = Tex->GetName().GetID();
		TextureNameList.insert({ ID, Tex->GetName().GetDisplayName() });
		TextureList.insert({ ID, Tex });
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
	
	RTexturePtr TextureManager::CreateTexture2D(const WString & Name, const WString & Origin,
		EPixelFormat Format, EFilterMode FilterMode, ESamplerAddressMode AddressMode, const IntVector2 & Size, bool GenMipMapsOnLoad) {
		RTexturePtr Texture = GetTexture(Name);
		if (Texture == NULL) {
			Texture = RTexturePtr(new RTexture(
				Name, Origin, ETextureDimension::Texture2D, Format, FilterMode, AddressMode, IntVector3(Size.x, Size.y, 1), GenMipMapsOnLoad
			));
			AddTexture(Texture);
		}
		return Texture;
	}
	
	RTexturePtr TextureManager::CreateCubemap(const WString & Name, const WString & Origin,
		EPixelFormat Format, EFilterMode FilterMode, ESamplerAddressMode AddressMode, const int & Size) {
		RTexturePtr Texture = GetTexture(Name);
		if (Texture == NULL) {
			Texture = RTexturePtr(new RTexture(
				Name, Origin, ETextureDimension::Cubemap, Format, FilterMode, AddressMode, IntVector3(Size, Size, 6)
			));
			AddTexture(Texture);
		}
		return Texture;
	}

	void TextureManager::LoadImageFromFile(
		const WString& Name, EPixelFormat ColorFormat, EFilterMode FilterMode,
		ESamplerAddressMode AddressMode, bool bFlipVertically, bool bGenMipMaps, const WString & FilePath, bool bConservePixels) {

		RTexturePtr LoadedTexture = CreateTexture2D(Name, FilePath, ColorFormat, FilterMode, AddressMode);
		
		if (LoadedTexture) {
			LoadedTexture->SetGenerateMipMapsOnLoad(bGenMipMaps);
			LoadedTexture->Load();
			if (!bConservePixels) LoadedTexture->ClearPixelData();
		}
	}

	TextureManager & TextureManager::GetInstance() {
		static TextureManager Manager;
		return Manager;
	}

}
