#pragma once

#include "CoreMinimal.h"
#include "Files/FileManager.h"
#include "Utility/Hasher.h"

namespace ESource {

	class OBJLoader {
	private:
		struct ObjectData {
			struct Subdivision {
				NString Name;
				int VertexIndicesPos = 0;
				int VertexIndicesCount = 0;
			};

			NString Name;
			Box3D Bounding;
			TArray<Subdivision> Materials;
			bool hasNormals;
			bool hasTextureCoords;
			int VertexIndicesPos = 0;
			int VertexIndicesCount = 0;
		};

		struct ExtractedData {
			TArray<ObjectData> Objects;
			TArray<IntVector3> VertexIndices;
			MeshVector3D ListPositions;
			MeshVector3D ListNormals;
			MeshUVs ListUVs;
			TArray<const NChar *> LineVertexIndices;
			TArray<const NChar *> LinePositions;
			TArray<const NChar *> LineNormals;
			TArray<const NChar *> LineUVs;
		};

		enum Keyword {
			Comment, Object, Group, Material, Vertex, Normal, TextureCoord, Face, CSType, Undefined
		};

		static Keyword GetKeyword(const NChar* Line);

		static bool GetSimilarVertexIndex(
			const StaticVertex & Vertex,
			TDictionary<StaticVertex, unsigned> & VertexToIndex,
			unsigned & Result
		);

		static void ExtractVector3(const NChar * Text, Vector3* Vertex);
		static void ExtractVector2(const NChar * Text, Vector2* Vertex);
		static void ExtractIntVector3(const NChar * Text, IntVector3* Vertex);

		static void PrepareData(const NChar * InFile, ExtractedData& Data);

		static void ParseVertexPositions(ExtractedData& Data);
		static void ParseVertexNormals(ExtractedData& Data);
		static void ParseVertexUVs(ExtractedData& Data);
		static void ParseFaces(ExtractedData& Data);

	public:
		/** Load mesh data from file extension Wavefront, it will return the models separated by objects, optionaly
		  * there's a way to optimize the vertices. */
		static bool LoadModel(ModelParser::ModelDataInfo & Info, const ModelParser::ParsingOptions & Options);

	};

}

MAKE_HASHABLE(ESource::Vector2, t.X, t.Y)
MAKE_HASHABLE(ESource::Vector3, t.X, t.Y, t.Z)
MAKE_HASHABLE(ESource::Vector4, t.X, t.Y, t.Z, t.W)
MAKE_HASHABLE(ESource::StaticVertex, t.Position, t.Normal, t.Tangent, t.UV0, t.UV1, t.Color)