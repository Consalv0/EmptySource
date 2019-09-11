
#include "CoreMinimal.h"
#include "Resources/MaterialManager.h"

namespace EmptySource {

	MaterialPtr MaterialManager::GetMaterial(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetMaterial(UID);
	}

	MaterialPtr MaterialManager::GetMaterial(const size_t & UID) const {
		auto Resource = MaterialList.find(UID);
		if (Resource != MaterialList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void MaterialManager::FreeMaterial(const WString & Name) {
		size_t UID = WStringToHash(Name);
		MaterialNameList.erase(UID);
		MaterialList.erase(UID);
	}

	void MaterialManager::AddMaterial(const WString & Name, MaterialPtr Material) {
		size_t UID = WStringToHash(Name);
		MaterialNameList.insert({ UID, Name });
		MaterialList.insert({ UID, Material });
	}

	TArray<WString> MaterialManager::GetResourceNames() const {
		TArray<WString> Names;
		for (auto KeyValue : MaterialNameList)
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

	void MaterialManager::LoadResourcesFromFile(const WString & FilePath) {
	}

	MaterialManager & MaterialManager::GetInstance() {
		static MaterialManager Manager;
		return Manager;
	}

}