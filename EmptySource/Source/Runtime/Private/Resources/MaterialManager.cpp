
#include "CoreMinimal.h"
#include "Resources/MaterialManager.h"

namespace ESource {

	MaterialPtr MaterialManager::GetMaterial(const IName & Name) const {
		return GetMaterial(Name.GetID());
	}

	MaterialPtr MaterialManager::GetMaterial(const size_t & UID) const {
		auto Resource = MaterialList.find(UID);
		if (Resource != MaterialList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	void MaterialManager::FreeMaterial(const IName & Name) {
		size_t UID = Name.GetID();
		MaterialNameList.erase(UID);
		MaterialList.erase(UID);
	}

	void MaterialManager::AddMaterial(MaterialPtr Material) {
		size_t UID = Material->GetName().GetID();
		MaterialNameList.insert({ UID, Material->GetName() });
		MaterialList.insert({ UID, Material });
	}

	MaterialPtr MaterialManager::CreateMaterial(
		const IName & Name, const RShaderPtr & Shader, bool WriteDepth, EDepthFunction DepthFunction,
		ERasterizerFillMode FillMode, ECullMode CullMode, TArrayInitializer<ShaderParameter> Parameters) 
	{
		MaterialPtr NewMaterial = std::make_shared<Material>(Name);
		NewMaterial->bWriteDepth = WriteDepth;
		NewMaterial->DepthFunction = DepthFunction;
		NewMaterial->FillMode = FillMode;
		NewMaterial->CullMode = CullMode;
		NewMaterial->SetShaderProgram(Shader);
		NewMaterial->SetParameters(Parameters);
		AddMaterial(NewMaterial);
		return NewMaterial;
	}

	TArray<IName> MaterialManager::GetResourceNames() const {
		TArray<IName> Names;
		for (auto KeyValue : MaterialNameList)
			Names.push_back(KeyValue.second);
		std::sort(Names.begin(), Names.end(), [](const IName& First, const IName& Second) {
			return First < Second;
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