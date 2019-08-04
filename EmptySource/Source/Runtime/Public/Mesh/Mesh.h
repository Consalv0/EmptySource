#pragma once

#include "Engine/Text.h"
#include "Math/MathUtility.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/IntVector3.h"
#include "Math/Vector4.h"
#include "Math/Box3D.h"

namespace EmptySource {

	// Default vertex Locations
	constexpr auto PositionLocation = 0;
	constexpr auto NormalLocation = 1;
	constexpr auto TangentLocation = 2;
	constexpr auto UV0Location = 3;
	constexpr auto UV1Location = 4;
	constexpr auto ColorLocation = 5;
	constexpr auto WeightsLocation = 9;

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
	typedef TDictionary<String, int> MeshMaterials;

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
	private:
		struct ElementBuffer {
			unsigned int VertexArrayObject; unsigned int Buffer;
			void Clear();
			bool SetUpBuffers(const unsigned int & VertexBuffer, const MeshFaces & Indices);
			void Bind() const;
			bool IsValid() const;
		};

		//* Vertex Array Object 
		unsigned int VertexBuffer;

		ElementBuffer ElementBufferObject;
		TArray<ElementBuffer> ElementBufferSubdivisions;

	public:
		MeshData Data;

		Mesh();
		//* Copy the information to the mesh, the data will be duplicated
		Mesh(const MeshData & OtherData);
		//* Transfer information to the mesh, the data will be swapped
		Mesh(MeshData * OtherData);

		//* *Create or Bind Vertex Array Object
		void BindVertexArray() const;

		//* Bind Element Subdivision Array Object
		void BindSubdivisionVertexArray(int MaterialIndex) const;

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
	};

}