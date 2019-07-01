#pragma once

#include "../include/FileManager.h"
#include "../include/MeshLoader.h"
#include "../include/Core.h"
#include "../include/Utility/Hasher.h"

class OBJLoader {
private:
	struct ObjectData {
		String Name;
		Box3D Bounding;
		bool hasNormals;
		bool hasTextureCoords;
		int VertexIndicesPos = 0;
		int VertexIndicesCount = 0;
	};

	struct ExtractedData {
		TArray<String> Groups;
		TArray<ObjectData> Objects;
		TArray<IntVector3> VertexIndices;
		MeshVector3D ListPositions;
		MeshVector3D ListNormals;
		MeshUVs ListUVs;
		TArray<const Char *> LineVertexIndices;
		TArray<const Char *> LinePositions;
		TArray<const Char *> LineNormals;
		TArray<const Char *> LineUVs;
	};

	enum Keyword {
		Comment, Object, Group, Vertex, Normal, TextureCoord, Face, CSType, Undefined
	};

	static Keyword GetKeyword(const Char* Line);

	static bool GetSimilarVertexIndex(
		const MeshVertex & Vertex,
		TDictionary<MeshVertex, unsigned> & VertexToIndex,
		unsigned & Result
	);

	static void ExtractVector3(const Char * Text, Vector3* Vertex);
	static void ExtractVector2(const Char * Text, Vector2* Vertex);
	static void ExtractIntVector3(const Char * Text, IntVector3* Vertex);

	static void PrepareData(const Char * InFile, ExtractedData& Data);

	static void ParseVertexPositions(ExtractedData& Data);
	static void ParseVertexNormals(ExtractedData& Data);
	static void ParseVertexUVs(ExtractedData& Data);
	static void ParseFaces(ExtractedData& Data);

public:
	/** Load mesh data from file extension Wavefront, it will return the models separated by objects, optionaly
	  * there's a way to optimize the vertices. */
	static bool Load(MeshLoader::FileData & FileData);

};

MAKE_HASHABLE(Vector2, t.x, t.y)
MAKE_HASHABLE(Vector3, t.x, t.y, t.z)
MAKE_HASHABLE(Vector4, t.x, t.y, t.z, t.w)
MAKE_HASHABLE(MeshVertex, t.Position, t.Normal, t.Tangent, t.UV0, t.UV1, t.Color)