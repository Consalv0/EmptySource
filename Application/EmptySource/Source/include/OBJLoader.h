#pragma once

#include "../include/FileManager.h"
#include "../include/Mesh.h"
#include "../include/Core.h"

class OBJLoader {
private:
	struct ObjectData {
		String Name;
		Box3D Bounding;
		int VertexIndicesCount = 0;
		int PositionCount = 0;
		int NormalCount = 0;
		int UVsCount = 0;
	};

	struct FileData {
		TArray<ObjectData> Objects;
		TArray<IntVector3> VertexIndices;
		MeshVector3D ListPositions;
		MeshVector3D ListNormals;
		MeshUVs ListUVs;
		int VertexIndicesCount = 0;
		int PositionCount = 0;
		int NormalCount = 0;
		int UVsCount = 0;
	};

	enum Keyword {
		Comment, Object, Vertex, Normal, TextureCoord, Face, CSType, Undefined
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

	static void ComputeTangents(const MeshFaces & Faces, MeshVertices & Vertices);

	static void ReadLineByLine(
		const Char * InFile,
		FileData& FileData
	);

	static void PrepareData(
		const Char * InFile,
		FileData& FileData
	);

	static void ParseLine(
		const Keyword& Keyword,
		Char* Line,
		FileData& FileData,
		int ObjectCount
	);

public:
	/** Load mesh data from file extension Wavefront, it will return the models separated by objects, optionaly
	  * there's a way to optimize the vertices. */
	static bool Load(FileStream* File, TArray<MeshFaces>* Faces, TArray<MeshVertices>* Vertices, TArray<Box3D>* BoundingBoxes, bool Optimize = true);

};

inline void HashCombine(std::size_t& seed) { }

template <typename T, typename... Rest>
inline void HashCombine(std::size_t& seed, const T& v, Rest... rest) {
	std::hash<T> Hasher;
	seed ^= Hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	HashCombine(seed, rest...);
}

#define MAKE_HASHABLE(type, ...) \
    namespace std {\
        template<> struct hash<type> {\
            std::size_t operator()(const type &t) const {\
                std::size_t ret = 0;\
                HashCombine(ret, __VA_ARGS__);\
                return ret;\
            }\
        };\
    }

MAKE_HASHABLE(Vector2, t.x, t.y)
MAKE_HASHABLE(Vector3, t.x, t.y, t.z)
MAKE_HASHABLE(Vector4, t.x, t.y, t.z, t.w)
MAKE_HASHABLE(MeshVertex, t.Position, t.Normal, t.Tangent, t.UV0, t.UV1, t.Color)