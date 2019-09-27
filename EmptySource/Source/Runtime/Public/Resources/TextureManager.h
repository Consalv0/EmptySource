#pragma once

#include "Resources/ResourceManager.h"
#include "Resources/TextureResource.h"

namespace EmptySource {

	class TextureManager : public ResourceManager {
	public:
		RTexturePtr GetTexture(const IName& Name) const;

		RTexturePtr GetTexture(const size_t & UID) const;

		void FreeTexture(const IName& Name);

		void AddTexture(RTexturePtr Texture);
		
		TArray<WString> GetResourceNames() const;

		virtual inline EResourceType GetResourceType() const override { return RT_Texture; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		RTexturePtr CreateTexture2D(const WString& Name, const WString & Origin,
			EPixelFormat Format, EFilterMode FilterMode, ESamplerAddressMode AddressMode, const IntVector2 & Size = 0, bool bGenMipMapsOnLoad = false);

		RTexturePtr CreateCubemap(const WString& Name, const WString & Origin,
			EPixelFormat Format, EFilterMode FilterMode, ESamplerAddressMode AddressMode, const int & Size = 0);

		void TextureManager::LoadImageFromFile(
			const WString& Name, EPixelFormat ColorFormat, EFilterMode FilterMode,
			ESamplerAddressMode AddressMode, bool bFlipVertically, bool bGenMipMaps, const WString & FilePath, bool bConservePixels = false
		);

		static TextureManager& GetInstance();

		inline TDictionary<size_t, RTexturePtr>::iterator begin() { return TextureList.begin(); }
		inline TDictionary<size_t, RTexturePtr>::iterator end() { return TextureList.end(); }

	private:
		TDictionary<size_t, WString> TextureNameList;
		TDictionary<size_t, RTexturePtr> TextureList;

	};

}