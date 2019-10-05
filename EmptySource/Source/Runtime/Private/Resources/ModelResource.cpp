
#include "CoreMinimal.h"
#include "Resources/ModelResource.h"
#include "Resources/ModelManager.h"

#include "Resources/ModelParser.h"

namespace ESource {
	
	bool RModel::IsValid() const {
		return LoadState == LS_Loaded && Meshes.size() > 0;
	}
	
	void RModel::Load() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			LOG_CORE_DEBUG(L"Loading Model {}...", Name.GetDisplayName().c_str());
			if (!Origin.empty()) {
				FileStream * ModelFile = FileManager::GetFile(Origin);
				if (ModelFile == NULL) {
					LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}
				ModelParser::ModelDataInfo Info;
				ModelParser::ParsingOptions Options = { ModelFile, bOptimizeOnLoad };
				ModelParser::Load(Info, Options);

				for (TArray<MeshData>::iterator Data = Info.Meshes.begin(); Data != Info.Meshes.end(); ++Data) {
					auto LoadedMesh = ModelManager::GetInstance().CreateSubModelMesh(Name, *Data);
					Meshes.emplace(LoadedMesh->GetName().GetID(), LoadedMesh);
					for (auto & Mat : Data->Materials) {
						DefaultMaterials.try_emplace(Mat.second, Material(Text::NarrowToWide(Mat.second)));
					}
					LoadedMesh->Load();
				}
			}
		}
		LoadState = Meshes.size() > 0 ? LS_Loaded : LS_Unloaded;
	}
	
	void RModel::LoadAsync() {
		if (LoadState == LS_Loaded || LoadState == LS_Loading) return;

		LoadState = LS_Loading;
		{
			LOG_CORE_DEBUG(L"Loading Model {}...", Name.GetDisplayName().c_str());
			if (!Origin.empty()) {
				FileStream * ModelFile = FileManager::GetFile(Origin);
				if (ModelFile == NULL) {
					LOG_CORE_ERROR(L"Error reading file for shader: '{}'", Origin);
					LoadState = LS_Unloaded;
					return;
				}

				ModelParser::ParsingOptions Options = { ModelFile, bOptimizeOnLoad };
				ModelParser::LoadAsync(Options, [this](ModelParser::ModelDataInfo & Info) {
					for (TArray<MeshData>::iterator Data = Info.Meshes.begin(); Data != Info.Meshes.end(); ++Data) {
						auto LoadedMesh = ModelManager::GetInstance().CreateSubModelMesh(Name, *Data);
						Meshes.emplace(LoadedMesh->GetName().GetID(), LoadedMesh);
						for (auto & Mat : Data->Materials) {
							DefaultMaterials.try_emplace(Mat.second, Material(Text::NarrowToWide(Mat.second)));
						}
						LoadedMesh->Load();
					}

					LoadState = Meshes.size() > 0 ? LS_Loaded : LS_Unloaded;
				});
			}
		}
	}
	
	void RModel::Unload() {
		if (LoadState == LS_Unloaded || LoadState == LS_Unloading) return;

		LoadState = LS_Unloading;
		Meshes.clear();
		DefaultMaterials.clear();
		LoadState = LS_Unloaded;
	}
	
	void RModel::Reload() {
		Unload();
		Load();
	}

	inline size_t RModel::GetMemorySize() const {
		return size_t();
	}

	RModel::RModel(const IName & Name, const WString & Origin, bool bOptimize) 
		: ResourceHolder(Name, Origin), Meshes(), DefaultMaterials(), bOptimizeOnLoad(bOptimize) {
	}

	RModel::~RModel() {
		Unload();
	}

}