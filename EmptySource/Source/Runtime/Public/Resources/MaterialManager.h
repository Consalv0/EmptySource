#pragma once

#include "Resources/ResourceManager.h"
#include "Rendering/Material.h"

namespace EmptySource {

	class MaterialManager : public ResourceManager {
	public:
		MaterialPtr GetMaterial(const IName& Name) const;

		MaterialPtr GetMaterial(const size_t & UID) const;

		void FreeMaterial(const IName& Name);

		void AddMaterial(MaterialPtr Material);

		TArray<IName> GetResourceNames() const;

		virtual inline EResourceType GetResourceType() const override { return RT_Material; };

		virtual void LoadResourcesFromFile(const WString& FilePath) override;

		static MaterialManager& GetInstance();

		inline TDictionary<size_t, MaterialPtr>::iterator begin() { return MaterialList.begin(); }
		inline TDictionary<size_t, MaterialPtr>::iterator end() { return MaterialList.end(); }

	private:
		TDictionary<size_t, IName> MaterialNameList;
		TDictionary<size_t, MaterialPtr> MaterialList;

	};

}