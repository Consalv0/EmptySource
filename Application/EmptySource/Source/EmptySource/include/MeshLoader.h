#pragma once

#include "..\include\FileManager.h"
#include "..\include\Mesh.h"
#include "..\include\Core.h"

class MeshLoader {
public:

	struct OBJObjectData {
		String Name;
		int VertexIndicesCount = 0;
		int PositionCount = 0;
		int NormalCount = 0;
		int UVsCount = 0;
	};

	struct OBJFileData{
		std::vector<OBJObjectData> Objects;
		std::vector<IntVector3> VertexIndices;
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
		OBJFileData& FileData
	); 
	static void PrepareOBJData(
		const Char * InFile,
		OBJFileData& FileData
	);
	static OBJKeyword GetOBJKeyword(const Char* Line);
	static void ParseOBJLine(
		const OBJKeyword& Keyword,
		Char* Line,
		OBJFileData& FileData,
		int ObjectCount
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

// Original crack_atof version is at http://crackprogramming.blogspot.sg/2012/10/implement-atof.html
// But it cannot convert floating point with high +/- exponent.
// The version below by Tian Bo fixes that problem and improves performance by 10%
// http://coliru.stacked-crooked.com/a/2e28f0d71f47ca5e
inline double StringToDouble(const char* String, char** Pointer) {

	*Pointer = (char*)String;
	if (!*Pointer || !**Pointer) {
		return 0;
	}

	int Sign = 1;
	double IntPart = 0.0;
	double FractionPart = 0.0;
	bool hasFraction = false;
	bool hasExpo = false;

	// Take care of +/- sign
	if (**Pointer == '-') {
		++*Pointer;
		Sign = -1;
	} else if (**Pointer == '+') {
		++*Pointer;
	}

	while (**Pointer != '\0' && **Pointer != ',' && **Pointer != ' ') {
		if (**Pointer >= '0' && **Pointer <= '9') {
			IntPart = IntPart * 10 + (**Pointer - '0');
		} else if (**Pointer == '.') {
			hasFraction = true;
			++*Pointer;
			break;
		} else if (**Pointer == 'e') {
			hasExpo = true;
			++*Pointer;
			break;
		} else {
			return Sign * IntPart;
		}
		++*Pointer;
	}

	if (hasFraction) {
		double fractionExpo = 0.1;

		while (**Pointer != '\0' && **Pointer != ',' && **Pointer != ' ') {
			if (**Pointer >= '0' && **Pointer <= '9') {
				FractionPart += fractionExpo * (**Pointer - '0');
				fractionExpo *= 0.1;
			} else if (**Pointer == 'e') {
				hasExpo = true;
				++*Pointer;
				break;
			} else {
				return Sign * (IntPart + FractionPart);
			}
			++*Pointer;
		}
	}

	// Parsing exponet part
	double expPart = 1.0;
	if ((**Pointer != '\0' && **Pointer != ',' && **Pointer != ' ') && hasExpo) {
		int ExponentSign = 1;
		if (**Pointer == '-') {
			ExponentSign = -1;
			++*Pointer;
		} else if (**Pointer == '+') {
			++*Pointer;
		}

		int e = 0;
		while ((**Pointer != '\0' && **Pointer != ',' && **Pointer != ' ') && **Pointer >= '0' && **Pointer <= '9') {
			e = e * 10 + **Pointer - '0';
			++*Pointer;
		}

		expPart = Pow10(ExponentSign * e);
	}
	++*Pointer;

	return Sign * (IntPart + FractionPart) * expPart;
}