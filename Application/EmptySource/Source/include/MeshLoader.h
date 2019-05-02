#pragma once

#include "../include/FileManager.h"
#include "../include/Mesh.h"

class MeshLoader {
private:
	static bool LoadFromFile(FileStream * File, TArray<MeshFaces>* Faces, TArray<MeshVertices>* Vertices, TArray<Box3D>* BoundingBoxes, bool Optimize);

public:
	static bool Load(TArray<Mesh> & Model, FileStream * File, bool Optimize);
};