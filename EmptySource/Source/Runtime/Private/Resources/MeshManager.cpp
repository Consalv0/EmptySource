
#include "CoreMinimal.h"
#include "Resources/MeshManager.h"
#include "Resources/MeshParser.h"

#include <future>
#include <thread>

namespace EmptySource {

	MeshPtr MeshManager::GetMesh(const WString & Name) const {
		size_t UID = WStringToHash(Name);
		return GetMesh(UID);
	}

	MeshPtr MeshManager::GetMesh(const size_t & UID) const {
		auto Resource = MeshList.find(UID);
		if (Resource != MeshList.end()) {
			return Resource->second;
		}

		return NULL;
	}

	TArray<WString> MeshManager::GetResourceNames() const {
		TArray<WString> Names;
		for (auto KeyValue : MeshList)
			Names.push_back(KeyValue.second->GetMeshData().Name);
		return Names;
	}

	void MeshManager::FreeMesh(const WString & Name) {
		size_t UID = WStringToHash(Name);
		MeshList.erase(UID);
	}

	void MeshManager::AddMesh(const WString & Name, MeshPtr Mesh) {
		size_t UID = WStringToHash(Name);
		MeshList.insert({ UID, Mesh });
	}

	void MeshManager::LoadResourcesFromFile(const WString & FilePath) {
	}

	void MeshManager::LoadFromFile(const WString & FilePath, bool bOptimize) {
		MeshParser::ResourceData ModelData(FileManager::GetFile(FilePath), bOptimize);
		MeshParser::Load(ModelData);
		for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
			auto LoadedMesh = std::make_shared<Mesh>(&(*Data));
			AddMesh(LoadedMesh->GetMeshData().Name, LoadedMesh);
			LoadedMesh->SetUpBuffers();
		}
	}

	void MeshManager::LoadAsyncFromFile(const WString & FilePath, bool bOptimize) {
		MeshParser::LoadAsync(FileManager::GetFile(FilePath), true, [this](MeshParser::ResourceData & ModelData) {
			for (TArray<MeshData>::iterator Data = ModelData.Meshes.begin(); Data != ModelData.Meshes.end(); ++Data) {
				auto LoadedMesh = std::make_shared<Mesh>(&(*Data));
				AddMesh(LoadedMesh->GetMeshData().Name, LoadedMesh);
				LoadedMesh->SetUpBuffers();
			}
		});
	}

	MeshManager & MeshManager::GetInstance() {
		static MeshManager Manager;
		return Manager;
	}

}