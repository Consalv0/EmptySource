#pragma once

#include "..\include\FileManager.h"
#include "..\include\Mesh.h"

#include "..\include\Core.h"

class MeshLoader {
public:
	static bool GetSimilarVertexIndex(
		MeshVertex & Vertex,
		std::map<MeshVertex, unsigned> & VertexToIndex,
		unsigned & Result
	);

	static bool FromOBJ(FileStream* File, MeshFaces* Faces, MeshVertices* Vertices);
};