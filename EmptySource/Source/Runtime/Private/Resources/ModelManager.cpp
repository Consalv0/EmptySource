
#include "CoreMinimal.h"
#include "Resources/ModelManager.h"
#include "Resources/ModelParser.h"

#include <future>
#include <thread>

namespace ESource {

	void ModelManager::LoadResourcesFromFile(const WString & FilePath) {
	}

	void ModelManager::LoadFromFile(const WString & FilePath, bool bOptimize) {
		FileStream * File = FileManager::GetFile(FilePath);
		RModelPtr Model = CreateModel(File->GetFileNameWithoutExtension(), FilePath, bOptimize);
		Model->Load();
	}

	void ModelManager::LoadAsyncFromFile(const WString & FilePath, bool bOptimize) {
		FileStream * File = FileManager::GetFile(FilePath);
		RModelPtr Model = CreateModel(File->GetFileNameWithoutExtension(), FilePath, bOptimize);
		Model->LoadAsync();
	}

	RModelPtr ModelManager::GetModel(const IName & Name) const {
		return GetModel(Name.GetID());
	}

	RModelPtr ModelManager::GetModel(const size_t & UID) const {
		auto Resource = ModelList.find(UID);
		if (Resource != ModelList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	TArray<IName> ModelManager::GetResourceModelNames() const {
		TArray<IName> Names;
		for (auto KeyValue : ModelNameList)
			Names.push_back(KeyValue.second);
		std::sort(Names.begin(), Names.end(), [](const IName& First, const IName& Second) {
			return First < Second;
		});
		return Names;
	}

	void ModelManager::FreeModel(const IName & Name) {
		size_t UID = Name.GetID();
		ModelNameList.erase(UID);
		ModelList.erase(UID);
	}

	void ModelManager::AddModel(const RModelPtr & Model) {
		size_t UID = Model->GetName().GetID();
		ModelNameList.insert({ UID, Model->GetName() });
		ModelList.insert({ UID, Model });
	}

	RMeshPtr ModelManager::GetMesh(const IName & Name) const {
		return GetMesh(Name.GetID());
	}

	RMeshPtr ModelManager::GetMesh(const size_t & UID) const {
		auto Resource = MeshList.find(UID);
		if (Resource != MeshList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	TArray<IName> ModelManager::GetResourceMeshNames() const {
		TArray<IName> Names;
		for (auto KeyValue : MeshNameList)
			Names.push_back(KeyValue.second);
		std::sort(Names.begin(), Names.end(), [](const IName& First, const IName& Second) {
			return First < Second;
		});
		return Names;
	}

	void ModelManager::FreeMesh(const IName & Name) {
		size_t UID = Name.GetID();
		MeshNameList.erase(UID);
		MeshList.erase(UID);
	}

	void ModelManager::AddMesh(RMeshPtr Mesh) {
		size_t UID = Mesh->GetName().GetID();
		MeshNameList.insert({ UID, Mesh->GetName() });
		MeshList.insert({ UID, Mesh });
	}

	RModelPtr ModelManager::CreateModel(const IName & Name, const WString & Origin, bool bOptimize) {
		RModelPtr Model = GetModel(Name);
		if (Model == NULL) {
			Model = RModelPtr(new RModel( Name, Origin, bOptimize ));
			AddModel(Model);
		}
		return Model;
	}

	RMeshPtr ModelManager::CreateSubModelMesh(const IName& ModelName, MeshData & Data) {
		IName Name = ModelName.GetDisplayName() + L":" + Text::NarrowToWide(Data.Name);
		RMeshPtr Mesh = GetMesh(Name);
		if (Mesh == NULL) {
			Mesh = RMeshPtr(new RMesh(Name, ModelName.GetDisplayName(), ModelName, Data));
			AddMesh(Mesh);
		}
		else if (Mesh->GetLoadState() != LS_Loaded || Mesh->GetLoadState() != LS_Loading) {
			Mesh->VertexData.Transfer(Data);
		}
		return Mesh;
	}

	RMeshPtr ModelManager::CreateSubModelMesh(const IName & ModelName, const WString & MeshName) {
		IName Name = ModelName.GetDisplayName() + L":" + MeshName;
		RMeshPtr Mesh = GetMesh(Name);
		if (Mesh == NULL) {
			Mesh = RMeshPtr(new RMesh(Name, ModelName.GetDisplayName(), ModelName));
			AddMesh(Mesh);
		}
		return Mesh;
	}

	RMeshPtr ModelManager::CreateMesh(MeshData & Data) {
		RMeshPtr Mesh = GetMesh(Text::NarrowToWide(Data.Name));
		if (Mesh == NULL) {
			Mesh = RMeshPtr(new RMesh(Text::NarrowToWide(Data.Name), L"", L"", Data));
			AddMesh(Mesh);
		}
		return Mesh;
	}

	ModelManager & ModelManager::GetInstance() {
		static ModelManager Manager;
		return Manager;
	}

}