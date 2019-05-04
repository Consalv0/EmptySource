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
	static bool LoadScene(class FbxScene * Scene, FileStream* File);
	
	static void ExtractVertexData(class FbxMesh * pMesh, TArray<IntVector3> & Faces, TArray<MeshVertex> & Vertices, Box3D & BoundingBox);
	static void ExtractTextureCoords(
		class FbxMesh * pMesh, MeshVertex & Vertex,
		const int & ControlPointIndex, const int & PolygonIndex, const int & PolygonVertexIndex
	);
	static void ExtractNormal(
		class FbxMesh * pMesh, MeshVertex & Vertex,
		const int & ControlPointIndex, const int & VertexIndex
	);
	static void ExtractVertexColor(
		class FbxMesh * pMesh, MeshVertex & Vertex,
		const int & ControlPointIndex, const int & VertexIndex
	);
	static bool ExtractTangent(
		class FbxMesh * pMesh, MeshVertex & Vertex,
		const int & ControlPointIndex, const int & VertexIndex
	);

	static void ComputeTangents(const MeshFaces & Faces, MeshVertices & Vertices);

public:
	/** Load mesh data from FBX, it will return the models separated by objects, optionaly
	  * there's a way to optimize the vertices. */
	static bool Load(FileStream* File, TArray<MeshFaces>* Faces, TArray<MeshVertices>* Vertices, TArray<Box3D>* BoundingBoxes, bool Optimize = true);
};