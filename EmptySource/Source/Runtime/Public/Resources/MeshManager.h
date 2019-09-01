#pragma once

#include "Files/FileManager.h"
#include "Core/Transform.h"
#include "Rendering/Mesh.h"
#include "Resources/ResourceManager.h"

namespace EmptySource {

	struct MeshManager : public ResourceManager {
	public:
		MeshPtr GetMesh(const WString& Name) const;

		MeshPtr GetMesh(const size_t & UID) const;

		void FreeMesh(const WString& Name);

		void AddMesh(const WString& Name, MeshPtr Mesh);

		virtual void LoadResourcesFromFile(const WString& FilePath);

		void LoadFromFile(const WString& FilePath, bool bOptimize);

		void LoadAsyncFromFile(const WString& FilePath, bool bOptimize);

		virtual inline EResourceType GetResourceType() const { return RT_Mesh; };

		static MeshManager& GetInstance();

	private:
		TDictionary<size_t, MeshPtr> MeshList;
	};

}