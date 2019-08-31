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

		virtual inline EResourceType GetResourceType() const override { return RT_Texture; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		void TextureManager::LoadImageFromFile(
			const WString& Name, EColorFormat ColorFormat, EFilterMode FilterMode,
			ESamplerAddressMode AddressMode, bool bFlipVertically, bool bGenMipMaps, const WString & FilePath
		);

		static TextureManager& GetInstance();

	private:
		TDictionary<size_t, TexturePtr> TextureList;

	};

}