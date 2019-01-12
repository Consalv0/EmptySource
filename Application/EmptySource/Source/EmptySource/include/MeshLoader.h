#pragma once

#include "..\include\FileManager.h"
#include "..\include\Mesh.h"
#include "..\include\Core.h"

class MeshLoader {
public:

	struct ParseData {
		std::vector<IntVector3> VertexIndices;
		int VertexIndicesCount = 0;
		MeshVector3D ListPositions;
		int PositionCount = 0;
		MeshVector3D ListNormals;
		int NormalCount = 0;
		MeshUVs ListUVs;
		int UVsCount = 0;
	};
	enum OBJKeyword {
		Comment, Object, Vertex, Normal, TextureCoord, Face, CSType, Undefined
	};

	static std::locale Locale;
	
	static bool GetSimilarVertexIndex(
		const MeshVertex & Vertex,
		std::unordered_map<MeshVertex, unsigned> & VertexToIndex,
		unsigned & Result
	);

	static void ExtractVector3(const Char * Text, Vector3* Vertex);
	static void ExtractVector2(const Char * Text, Vector2* Vertex);
	static void ExtractIntVector3(const Char * Text, IntVector3* Vertex);
	static void ReadOBJByLine(
		const Char * InFile,
		ParseData& Data
	); 
	static void PrepareOBJData(
		const Char * InFile,
		ParseData& Data
	);
	static OBJKeyword GetOBJKeyword(const Char* Line);
	static void ParseOBJLine(
		const OBJKeyword& Keyword,
		Char* Line,
		ParseData& Data
	);

	static bool FromOBJ(FileStream* File, MeshFaces* Faces, MeshVertices* Vertices, bool Optimize = true);
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