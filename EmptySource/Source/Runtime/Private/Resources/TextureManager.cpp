
#include "CoreMinimal.h"
#include "Resources/TextureManager.h"
#include "Resources/ImageConversion.h"

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
		TextureNameList.erase(UID);
		TextureList.erase(UID);
	}

	void TextureManager::AddTexture(const WString& Name, TexturePtr Tex) {
		size_t UID = WStringToHash(Name);
		TextureNameList.insert({ UID, Name });
		TextureList.insert({ UID, Tex });
	}

	TArray<WString> TextureManager::GetResourceNames() const {
		TArray<WString> Names;
		for (auto KeyValue : TextureNameList)
			Names.push_back(KeyValue.second);
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

		EColorFormat InColorFormat = ImageConversion::GetColorFormat(FileManager::GetFile(FilePath));
		Texture2DPtr LoadedTexture = NULL;
		switch (InColorFormat) {
		case EmptySource::CF_Red: {
			Bitmap<UCharRed> Bitmap;
			ImageConversion::LoadFromFile(Bitmap, FileManager::GetFile(FilePath), bFlipVertically);
			LoadedTexture = Texture2D::Create(Bitmap.GetSize(), ColorFormat, FilterMode, AddressMode, InColorFormat, Bitmap.PointerToValue());
		} break;
		case EmptySource::CF_RG: {
			Bitmap<UCharRG> Bitmap;
			ImageConversion::LoadFromFile(Bitmap, FileManager::GetFile(FilePath), bFlipVertically);
			LoadedTexture = Texture2D::Create(Bitmap.GetSize(), ColorFormat, FilterMode, AddressMode, InColorFormat, Bitmap.PointerToValue());
		} break;
		case EmptySource::CF_RGB: {
			Bitmap<UCharRGB> Bitmap;
			ImageConversion::LoadFromFile(Bitmap, FileManager::GetFile(FilePath), bFlipVertically);
			LoadedTexture = Texture2D::Create(Bitmap.GetSize(), ColorFormat, FilterMode, AddressMode, InColorFormat, Bitmap.PointerToValue());
		} break;
		case EmptySource::CF_RGBA: {
			Bitmap<UCharRGBA> Bitmap;
			ImageConversion::LoadFromFile(Bitmap, FileManager::GetFile(FilePath), bFlipVertically);
			LoadedTexture = Texture2D::Create(Bitmap.GetSize(), ColorFormat, FilterMode, AddressMode, InColorFormat, Bitmap.PointerToValue());
		} break;
		case EmptySource::CF_RGBA32F: {
			Bitmap<FloatRGBA> Bitmap;
			ImageConversion::LoadFromFile(Bitmap, FileManager::GetFile(FilePath), bFlipVertically);
			LoadedTexture = Texture2D::Create(Bitmap.GetSize(), ColorFormat, FilterMode, AddressMode, InColorFormat, Bitmap.PointerToValue());
		} break;
		case EmptySource::CF_RGB32F: {
			Bitmap<FloatRGB> Bitmap;
			ImageConversion::LoadFromFile(Bitmap, FileManager::GetFile(FilePath), bFlipVertically);
			LoadedTexture = Texture2D::Create(Bitmap.GetSize(), ColorFormat, FilterMode, AddressMode, InColorFormat, Bitmap.PointerToValue());
		} break;
		default:
			ES_CORE_ASSERT(true, "Color format not implemented");
			break;
		}

		if (LoadedTexture) {
			if (bGenMipMaps) LoadedTexture->GenerateMipMaps();
			AddTexture(Name, LoadedTexture);
		}
	}

	TextureManager & TextureManager::GetInstance() {
		static TextureManager Manager;
		return Manager;
	}

}
