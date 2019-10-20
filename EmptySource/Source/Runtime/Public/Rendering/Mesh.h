#pragma once

#include "Rendering/Rendering.h"

namespace ESource {

	struct StaticVertex {
		Vector3 Position;
		Vector3 Normal;
		Vector3 Tangent;
		Vector2 UV0, UV1;
		Vector4 Color;

		StaticVertex() = default;
		StaticVertex(const StaticVertex& Other) = default;
		StaticVertex(StaticVertex&& Other) = default;
		StaticVertex(const Vector3& P, const Vector3& N, const Vector2& UV);
		StaticVertex(const Vector3& P, const Vector3& N, const Vector3& T, const Vector2& UV0, const Vector2& UV1, const Vector4& C);
		StaticVertex& operator=(const StaticVertex& other) = default;
		bool operator<(const StaticVertex That) const;
		bool operator==(const StaticVertex &Other) const;
	};

	struct SkinVertex {
		Vector3 Position;
		Vector3 Normal;
		Vector3 Tangent;
		Vector2 UV0, UV1;
		Vector4 Color;

		uint32_t InfluenceBones[4] = { 0, 0, 0, 0 };
		float Weights[4]{ 0.0F, 0.0F, 0.0F, 0.0F };

		void AddBoneData(uint32_t BoneID, float Weight) {
			for (size_t i = 0; i < 4; i++) {
				if (Weights[i] == 0.0) {
					InfluenceBones[i] = BoneID;
					Weights[i] = Weight;
					return;
				}
			}
			LOG_CORE_WARN("Vertex has more than four bones/weights affecting it, extra data will be discarded (BoneID={0}, Weight={1})", BoneID, Weight);
		}

		SkinVertex() = default;
		SkinVertex(const SkinVertex& Other) = default;
		SkinVertex(SkinVertex&& Other) = default;
		SkinVertex(const StaticVertex& Other);
		SkinVertex& operator=(const SkinVertex& Other) = default;
	};

	struct Subdivision {
	public:
		uint32_t MaterialIndex;
		uint32_t BaseVertex;
		uint32_t BaseIndex;
		uint32_t IndexCount;
	};

	using FaceIndex                 = IntVector3;
	typedef TArray<FaceIndex>         MeshFaces;
	typedef TArray<Vector3>           MeshVector3D;
	typedef TArray<Vector2>           MeshUVs;
	typedef TArray<Vector4>           MeshColors;
	typedef TArray<StaticVertex>      MeshVertices;
	typedef TDictionary<int, NString> MeshMaterials;

	struct MeshData {
		NString Name;
		TArray<FaceIndex> Faces;
		TDictionary<int, Subdivision> SubdivisionsMap;
		TDictionary<int, NString> MaterialsMap;
		TArray<StaticVertex> StaticVertices;
		TArray<SkinVertex> SkinVertices;
		BoundingBox3D Bounding;

		bool hasNormals;
		bool hasTangents;
		bool hasVertexColor;
		int  UVChannels;
		bool hasBoundingBox;
		bool hasBones;

		void Transfer(MeshData & Other);
		void ComputeBounding();
		void ComputeTangents();
		void ComputeNormals();
		void Clear();
	};

	class Mesh {
	public:
		Mesh();
		//* Copy the information to the mesh, the data will be duplicated
		Mesh(const MeshData & OtherData);
		//* Transfer information to the mesh, the data will be swapped
		Mesh(MeshData * OtherData);

		inline const MeshData& GetMeshData() const { return Data; };

		//* Swap all contents of the mesh for new data
		void SwapMeshData(MeshData & Data);

		//* Copy all contents of the mesh for new data
		void CopyMeshData(const MeshData & Data);

		//* Get VertexArray in Mesh
		VertexArrayPtr GetVertexArray() const { return VertexArrayPointer; };

		//* Clear the mesh entirely
		void Clear();

		//* Clear the GL's objects
		void ClearBuffers();

		//* Give Vertices to OpenGL **This must be done once per render**
		bool SetUpBuffers();

	private:
		VertexArrayPtr VertexArrayPointer;

		MeshData Data;
	};

}