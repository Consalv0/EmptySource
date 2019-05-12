#pragma once

#include "../include/FileManager.h"
#include "../include/MeshLoader.h"
#include "../include/Core.h"

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

	struct FileData {
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

	static void PrepareData(const Char * InFile, FileData& Data);

	static void ParseVertexPositions(FileData& Data);
	static void ParseVertexNormals(FileData& Data);
	static void ParseVertexUVs(FileData& Data);
	static void ParseFaces(FileData& Data);

public:
	/** Load mesh data from file extension Wavefront, it will return the models separated by objects, optionaly
	  * there's a way to optimize the vertices. */
	static bool Load(FileStream* File, MeshLoader::FileData & OutData, bool Optimize = true);

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