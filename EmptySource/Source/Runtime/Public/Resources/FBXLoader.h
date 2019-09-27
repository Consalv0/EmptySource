#pragma once

#include "CoreMinimal.h"
#include "Files/FileManager.h"

namespace ESource {

	class FBXLoader {
	private:
		static class FbxManager * gSdkManager;

		//* Creates an importer object, and uses it to import a file into a scene.
		static bool LoadScene(class FbxScene * Scene, const FileStream* File);

		static void ExtractVertexData(class FbxMesh * pMesh, MeshData & OutData);
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
		static int ExtractMaterialIndex(
			class FbxMesh * pMesh, const int & PolygonIndex
		);
		static bool ExtractTangent(
			class FbxMesh * pMesh, MeshVertex & Vertex,
			const int & ControlPointIndex, const int & VertexIndex
		);

	public:

		//* Creates an instance of the SDK manager.
		static bool InitializeSdkManager();

		/** Load mesh data from FBX, it will return the models separated by objects, optionaly
		  * there's a way to optimize the vertices. */
		static bool Load(MeshParser::ResourceData & ResourceData);
	};

}