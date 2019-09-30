#pragma once

#include "Files/FileManager.h"
#include "Core/Transform.h"
#include "Resources/ModelParser.h"
#include "Resources/ModelResource.h"
#include "Resources/ResourceManager.h"

namespace ESource {

	class ModelManager : public ResourceManager {
	public:
		RModelPtr GetModel(const IName& Name) const;

		RModelPtr GetModel(const size_t & UID) const;

		TArray<IName> GetResourceModelNames() const;

		void FreeModel(const IName& Name);

		void AddModel(const RModelPtr & Model);

		RMeshPtr GetMesh(const IName& Name) const;

		RMeshPtr GetMesh(const size_t & UID) const;

		TArray<IName> GetResourceMeshNames() const;

		RModelPtr CreateModel(const IName& Name, const WString& Origin, bool bOptimize);

		RMeshPtr CreateSubModelMesh(const IName& ModelName, MeshData & Data);

		RMeshPtr CreateSubModelMesh(const IName& ModelName, const WString & MeshName);

		RMeshPtr CreateMesh(MeshData & Data);

		void FreeMesh(const IName& Name);

		void AddMesh(RMeshPtr Model);

		virtual void LoadResourcesFromFile(const WString& FilePath);

		void LoadFromFile(const WString& FilePath, bool bOptimize);

		void LoadAsyncFromFile(const WString& FilePath, bool bOptimize);

		virtual inline EResourceType GetResourceType() const { return RT_Model; };

		static ModelManager& GetInstance();

	private:		
		TDictionary<size_t, IName> ModelNameList;

		TDictionary<size_t, IName> MeshNameList;

		TDictionary<size_t, RMeshPtr> MeshList;

		TDictionary<size_t, RModelPtr> ModelList;
	};

}