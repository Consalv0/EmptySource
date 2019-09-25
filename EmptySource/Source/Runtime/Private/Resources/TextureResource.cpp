
#include "CoreMinimal.h"
#include "Resources/TextureResource.h"
#include "Resources/TextureManager.h"
#include "Resources/ImageConversion.h"

namespace EmptySource {
	
	RTexture::~RTexture() {
		Unload();
	}

	bool RTexture::IsValid() {
		return LoadState == LS_Loaded && TexturePointer->IsValid();
	}

	void RTexture::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			LOG_CORE_DEBUG(L"Loading Shader {}...", Name.GetDisplayName().c_str());
			if (!Origin.empty()) {
				FileStream * ShaderFile = FileManager::GetFile(Origin);
				if (ShaderFile == NULL) {
					LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}
				
				if (!ImageConversion::LoadFromFile(Pixels, ShaderFile, ColorFormat)) {
					ShaderFile->Close();
					LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}
				else {
					Dimensions = Pixels.GetDimensions();
					switch (Dimension) {
					case ETextureDimension::Texture2D:
						Texture2D::Create(Name.GetDisplayName(), Pixels.GetDimensions(), ColorFormat, FilterMode, AddressMode, ColorFormat, Pixels.PointerToValue());
					case ETextureDimension::Cubemap:
					case ETextureDimension::Texture1D:
					case ETextureDimension::Texture3D:
					default:
						break;
					}
				}
			}
		}
		LoadState = !Pixels.IsEmpty() ? LS_Loaded : LS_Unloaded;
	}

	void RTexture::Unload() {
	}

	void RTexture::Reload() {
		Unload();
		Load();
	}

	float RTexture::GetAspectRatio() const {
		return (float)Size.x / (float)Size.y;
	}

	RTexture::RTexture(
		const IName & Name, const WString & Origin,
		ETextureDimension Dimension, EColorFormat Format, EFilterMode FilterMode, ESamplerAddressMode AddressMode
	) 
		: ResourceHolder(Name, Origin), Dimension(Dimension), FilterMode(FilterMode), AddressMode(AddressMode), ColorFormat(Format) {
	}

}