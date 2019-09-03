#pragma once

#include "Resources/ResourceManager.h"
#include "Rendering/Texture.h"

namespace EmptySource {

	class TextureManager : public ResourceManager {
	public:
		TexturePtr GetTexture(const WString& Name) const;

		TexturePtr GetTexture(const size_t & UID) const;

		void FreeTexture(const WString& Name);

		void AddTexture(const WString& Name, TexturePtr Texture);
		
		TArray<WString> GetResourceNames() const;

		virtual inline EResourceType GetResourceType() const override { return RT_Texture; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		void TextureManager::LoadImageFromFile(
			const WString& Name, EColorFormat ColorFormat, EFilterMode FilterMode,
			ESamplerAddressMode AddressMode, bool bFlipVertically, bool bGenMipMaps, const WString & FilePath
		);

		static TextureManager& GetInstance();

		inline TDictionary<size_t, TexturePtr>::iterator begin() { return TextureList.begin(); }
		inline TDictionary<size_t, TexturePtr>::iterator end() { return TextureList.end(); }

	private:
		TDictionary<size_t, WString> TextureNameList;
		TDictionary<size_t, TexturePtr> TextureList;

	};

}