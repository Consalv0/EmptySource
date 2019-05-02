
#include "../include/Core.h"
#include "../include/OBJLoader.h"
#include "../include/FBXLoader.h"
#include "../include/MeshLoader.h"

bool MeshLoader::LoadFromFile(FileStream * File, TArray<MeshFaces>* Faces, TArray<MeshVertices>* Vertices, TArray<Box3D>* BoundingBoxes, bool Optimize) {
	WString Extension = File->GetExtension();
	if (Text::CompareIgnoreCase(Extension, L"FBX")) {
		return FBXLoader::Load(File, Faces, Vertices, BoundingBoxes, Optimize);
	}
	if (Text::CompareIgnoreCase(Extension, L"OBJ")) {
		return OBJLoader::Load(File, Faces, Vertices, BoundingBoxes, Optimize);
	}

	return false;
}

bool MeshLoader::Load(TArray<Mesh> & Model, FileStream * File, bool Optimize) {
	if (File == NULL) return false;

	Debug::Log(Debug::LogInfo, L"Reading File Model '%ls'", File->GetShortPath().c_str());
	TArray<MeshFaces> Faces; TArray<MeshVertices> Vertices; TArray<Box3D> BBoxes;
	bool bNoError = LoadFromFile(File, &Faces, &Vertices, &BBoxes, Optimize);
	if (bNoError) {
		for (int MeshDataCount = 0; MeshDataCount < Faces.size(); ++MeshDataCount) {
			Model.push_back(Mesh(&Faces[MeshDataCount], &Vertices[MeshDataCount], BBoxes[MeshDataCount]));
		}
	}
	return bNoError;
}
