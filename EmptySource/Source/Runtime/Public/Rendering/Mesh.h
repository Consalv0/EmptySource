#pragma once

#include "Rendering/Rendering.h"

namespace EmptySource {

	struct MeshVertex {
		Vector3 Position;
		Vector3 Normal;
		Vector3 Tangent;
		Vector2 UV0, UV1;
		Vector4 Color;

		MeshVertex() = default;
		MeshVertex(const MeshVertex& Other) = default;
		MeshVertex(MeshVertex&& Other) = default;
		MeshVertex(const Vector3& P, const Vector3& N, const Vector2& UV);
		MeshVertex(const Vector3& P, const Vector3& N, const Vector3& T, const Vector2& UV0, const Vector2& UV1, const Vector4& C);
		MeshVertex& operator=(const MeshVertex& other) = default;
		bool operator<(const MeshVertex That) const;
		bool operator==(const MeshVertex &Other) const;
	};

	typedef TArray<IntVector3> MeshFaces;
	typedef TArray<Vector3>    MeshVector3D;
	typedef TArray<Vector2>    MeshUVs;
	typedef TArray<Vector4>    MeshColors;
	typedef TArray<MeshVertex> MeshVertices;
	typedef TDictionary<int, NString> MeshMaterials;

	struct MeshData {
		WString Name;
		MeshFaces Faces;
		TDictionary<int, MeshFaces> MaterialSubdivisions;
		MeshVertices Vertices;
		MeshMaterials Materials;
		BoundingBox3D Bounding;

		bool hasNormals;
		bool hasTangents;
		bool hasVertexColor;
		int TextureCoordsCount;
		bool hasBoundingBox;
		bool hasWeights;

		void Swap(MeshData & Other);
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

		//* Create or Bind Vertex Array Object
		void BindVertexArray() const;

		//* Bind Element Subdivision Array Object
		void BindSubdivisionVertexArray(int MaterialIndex) const;

		//* Get VertexArray in Mesh
		VertexArrayPtr GetSubdivisionVertexArray(int MaterialIndex) const;

		//* Draw mesh using instanciated Element Buffer
		void DrawInstanciated(int Count) const;

		//* Draw mesh using instanciated Element Buffer unsing material subdivision
		void DrawSubdivisionInstanciated(int Count, int MaterialIndex) const;

		//* Draw mesh using Element Buffer
		void DrawElement() const;

		//* Clear the mesh entirely
		void Clear();

		//* Clear the GL's objects
		void ClearBuffers();

		//* Give Vertices to OpenGL **This must be done once per render**
		bool SetUpBuffers();

	private:
		VertexArrayPtr VertexArrayObject;
		TArray<VertexArrayPtr> MeshSubdivisions;
		MeshData Data;
	};

	typedef std::shared_ptr<Mesh> MeshPtr;

}