#pragma once

#include "../include/FileManager.h"
#include "../include/Transform.h"
#include "../include/Mesh.h"

class MeshLoader {
public:
	struct FileData {
		TArray<MeshData> Meshes;
		TArray<Transform> MeshTransforms;

		//* The model data has been loaded
		bool bLoaded;
	};

private:
	static bool LoadFromFile(FileStream * File, FileData & Data, bool Optimize);

public:
	static bool Load(FileData & Data, FileStream * File, bool Optimize);
};