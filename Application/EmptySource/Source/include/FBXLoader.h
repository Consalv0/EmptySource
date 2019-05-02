#pragma once

#include "../include/Core.h"
#include "../include/FileManager.h"
#include "../include/Mesh.h"

class FBXLoader {
private:
	static class FbxManager * gSdkManager;

	//* Creates an instance of the SDK manager.
	static void InitializeSdkManager();

	//* Creates an importer object, and uses it to import a file into a scene.
	static bool LoadScene(class FbxScene * pScene, FileStream* File);

public:
	/** Load mesh data from FBX, it will return the models separated by objects, optionaly
	  * there's a way to optimize the vertices. */
	static bool Load(FileStream* File, TArray<MeshFaces>* Faces, TArray<MeshVertices>* Vertices, TArray<Box3D>* BoundingBoxes, bool Optimize = true);
};