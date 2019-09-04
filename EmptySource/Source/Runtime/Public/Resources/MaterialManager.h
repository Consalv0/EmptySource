#pragma once

#pragma once

#include "Resources/ResourceManager.h"
#include "Rendering/Material.h"

namespace EmptySource {

	class MaterialManager : public ResourceManager {
	public:
		Material GetMaterial(const WString& Name) const;

		Material GetMaterial(const size_t & UID) const;

		void FreeMaterial(const WString& Name);

		void AddMaterial(const WString& Name, Material Material);

		TArray<WString> GetResourceNames() const;

		virtual inline EResourceType GetResourceType() const override { return RT_Material; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		void MaterialManager::LoadImageFromFile(
			const WString& Name, EColorFormat ColorFormat, EFilterMode FilterMode,
			ESamplerAddressMode AddressMode, bool bFlipVertically, bool bGenMipMaps, const WString & FilePath
		);

		static MaterialManager& GetInstance();

		inline TDictionary<size_t, Material>::iterator begin() { return MaterialList.begin(); }
		inline TDictionary<size_t, Material>::iterator end() { return MaterialList.end(); }

	private:
		TDictionary<size_t, WString> MaterialNameList;
		TDictionary<size_t, Material> MaterialList;

	};

}