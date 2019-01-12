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

// Original crack_atof version is at http://crackprogramming.blogspot.sg/2012/10/implement-atof.html
// But it cannot convert floating point with high +/- exponent.
// The version below by Tian Bo fixes that problem and improves performance by 10%
// http://coliru.stacked-crooked.com/a/2e28f0d71f47ca5e
inline double pow10(int n) {
	double ret = 1.0;
	double r = 10.0;
	if (n < 0) {
		n = -n;
		r = 0.1;
	}

	while (n) {
		if (n & 1) {
			ret *= r;
		}
		r *= r;
		n >>= 1;
	}
	return ret;
}

inline double crack_strtof(const char* str, char** num) {

	*num = (char*)str;
	if (!*num || !**num) {
		return 0;
	}

	int sign = 1;
	double integerPart = 0.0;
	double fractionPart = 0.0;
	bool hasFraction = false;
	bool hasExpo = false;

	// Take care of +/- sign
	if (**num == '-') {
		++*num;
		sign = -1;
	} else if (**num == '+') {
		++*num;
	}

	while (**num != '\0' && **num != ',' && **num != ' ') {
		if (**num >= '0' && **num <= '9') {
			integerPart = integerPart * 10 + (**num - '0');
		} else if (**num == '.') {
			hasFraction = true;
			++*num;
			break;
		} else if (**num == 'e') {
			hasExpo = true;
			++*num;
			break;
		} else {
			return sign * integerPart;
		}
		++*num;
	}

	if (hasFraction) {
		double fractionExpo = 0.1;

		while (**num != '\0' && **num != ',' && **num != ' ') {
			if (**num >= '0' && **num <= '9') {
				fractionPart += fractionExpo * (**num - '0');
				fractionExpo *= 0.1;
			} else if (**num == 'e') {
				hasExpo = true;
				++*num;
				break;
			} else {
				return sign * (integerPart + fractionPart);
			}
			++*num;
		}
	}

	// parsing exponet part
	double expPart = 1.0;
	if ((**num != '\0' && **num != ',' && **num != ' ') && hasExpo) {
		int expSign = 1;
		if (**num == '-') {
			expSign = -1;
			++*num;
		} else if (**num == '+') {
			++*num;
		}

		int e = 0;
		while ((**num != '\0' && **num != ',' && **num != ' ') && **num >= '0' && **num <= '9') {
			e = e * 10 + **num - '0';
			++*num;
		}

		expPart = pow10(expSign * e);
	}
	++*num;

	return sign * (integerPart + fractionPart) * expPart;
}