#pragma once

#include "../include/FileManager.h"
#include "../include/Mesh.h"
#include "../include/Core.h"

class MeshLoader {
private:
	struct OBJObjectData {
		String Name;
		int VertexIndicesCount = 0;
		int PositionCount = 0;
		int NormalCount = 0;
		int UVsCount = 0;
	};

	struct OBJFileData {
		TArray<OBJObjectData> Objects;
		TArray<IntVector3> VertexIndices;
		MeshVector3D ListPositions;
		MeshVector3D ListNormals;
		MeshUVs ListUVs;
		int VertexIndicesCount = 0;
		int PositionCount = 0;
		int NormalCount = 0;
		int UVsCount = 0;
	};

	enum OBJKeyword {
		Comment, Object, Vertex, Normal, TextureCoord, Face, CSType, Undefined
	};

	static OBJKeyword GetOBJKeyword(const Char* Line);

	static bool GetSimilarVertexIndex(
		const MeshVertex & Vertex,
		TDictionary<MeshVertex, unsigned> & VertexToIndex,
		unsigned & Result
	);

	static void ExtractVector3(const Char * Text, Vector3* Vertex);
	static void ExtractVector2(const Char * Text, Vector2* Vertex);
	static void ExtractIntVector3(const Char * Text, IntVector3* Vertex);

	static size_t ReadOBJByLine(
		const Char * InFile,
		OBJFileData& FileData
	);

	static void PrepareOBJData(
		const Char * InFile,
		OBJFileData& FileData
	);

	static void ParseOBJLine(
		const OBJKeyword& Keyword,
		Char* Line,
		OBJFileData& FileData,
		int ObjectCount
	);

public:
	/** Load mesh data from file extension Wavefront, it will return the models separated by objects, optionaly
	  * there's a way to optimize the vertices. */
	static bool FromOBJ(FileStream* File, std::vector<MeshFaces>* Faces, std::vector<MeshVertices>* Vertices, bool Optimize = true);
};

inline void hash_combine(std::size_t& seed) { }

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	hash_combine(seed, rest...);
}

#define MAKE_HASHABLE(type, ...) \
    namespace std {\
        template<> struct hash<type> {\
            std::size_t operator()(const type &t) const {\
                std::size_t ret = 0;\
                hash_combine(ret, __VA_ARGS__);\
                return ret;\
            }\
        };\
    }

MAKE_HASHABLE(Vector2, t.x, t.y)
MAKE_HASHABLE(Vector3, t.x, t.y, t.z)
MAKE_HASHABLE(Vector4, t.x, t.y, t.z, t.w)
MAKE_HASHABLE(MeshVertex, t.Position, t.Normal, t.Tangent, t.UV0, t.UV1, t.Color)

