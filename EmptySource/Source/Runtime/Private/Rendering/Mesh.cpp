
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/Mesh.h"

namespace ESource {

	StaticVertex::StaticVertex(const Vector3 & P, const Vector3 & N, const Vector2 & UV) :
		Position(P), Normal(N), Tangent(), UV0(UV), UV1(UV), Color(1.0F) {
	}

	StaticVertex::StaticVertex(const Vector3 & P, const Vector3 & N, const Vector3 & T,
		const Vector2 & UV0, const Vector2 & UV1, const Vector4 & C) :
		Position(P), Normal(N), Tangent(T),
		UV0(UV0), UV1(UV1), Color(C) {
	}

	bool StaticVertex::operator<(const StaticVertex That) const {
		return memcmp((void*)this, (void*)&That, sizeof(StaticVertex)) > 0;
	}

	bool StaticVertex::operator==(const StaticVertex & Other) const {
		return (Position == Other.Position
			&& Normal == Other.Normal
			&& Tangent == Other.Tangent
			&& UV0 == Other.UV0
			&& UV1 == Other.UV1
			&& Color == Other.Color);
	};

	SkinVertex::SkinVertex(const StaticVertex & Other) {
		Position = Other.Position;
		Normal   = Other.Normal;
		Tangent  = Other.Tangent;
		UV0      = Other.UV0;
		UV1      = Other.UV1;
		Color    = Other.Color;
	}

	Mesh::Mesh() {
		VertexArrayPointer = NULL;
		Data = MeshData();
	}

	Mesh::Mesh(const MeshData & OtherData) {
		VertexArrayPointer = NULL;
		Data = OtherData;
	}

	Mesh::Mesh(MeshData * OtherData) {
		VertexArrayPointer = NULL;
		Data.Transfer(*OtherData);
	}

	void Mesh::SwapMeshData(MeshData & NewData) {
		Clear(); Data.Transfer(NewData);
	}

	void Mesh::CopyMeshData(const MeshData & NewData) {
		Clear(); Data = NewData;
	}

	bool Mesh::SetUpBuffers() {
		if (Data.StaticVertices.size() <= 0 || Data.Faces.size() <= 0) return false;

		static BufferLayout DafultLayout = {
			{ EShaderDataType::Float3, "_iVertexPosition" },
			{ EShaderDataType::Float3, "_iVertexNormal", true },
			{ EShaderDataType::Float3, "_iVertexTangent", true },
			{ EShaderDataType::Float2, "_iVertexUV0" },
			{ EShaderDataType::Float2, "_iVertexUV1" },
			{ EShaderDataType::Float4, "_iVertexColor" }
		};

		// Give our vertices to VAO
		VertexBufferPtr VertexBufferPointer = NULL;
		VertexBufferPointer = VertexBuffer::Create((float *)&Data.StaticVertices[0], (unsigned int)(Data.StaticVertices.size() * sizeof(StaticVertex)), UM_Static);
		VertexBufferPointer->SetLayout(DafultLayout);

		VertexArrayPointer = VertexArray::Create();
		IndexBufferPtr IndexBufferPointer = IndexBuffer::Create(
			(unsigned int *)&Data.Faces[0],
			(unsigned int)Data.Faces.size() * 3, UM_Static
		);
		VertexArrayPointer->AddVertexBuffer(VertexBufferPointer);
		VertexArrayPointer->AddIndexBuffer(IndexBufferPointer);
		VertexArrayPointer->Unbind();

		return true;
	}

	void Mesh::ClearBuffers() {
		VertexArrayPointer = NULL;
	}

	void Mesh::Clear() {
		Data.Clear();
		ClearBuffers();
	}

	void MeshData::Transfer(MeshData & Other) {
		Name = Other.Name;
		Faces.swap(Other.Faces);
		StaticVertices.swap(Other.StaticVertices);
		SkinVertices.swap(Other.SkinVertices);
		SubdivisionsMap.swap(Other.SubdivisionsMap);
		MaterialsMap.swap(Other.MaterialsMap);
		Bounding = Other.Bounding;

		hasNormals = Other.hasNormals;
		hasTangents = Other.hasTangents;
		hasVertexColor = Other.hasVertexColor;
		UVChannels = Other.UVChannels;
		hasBoundingBox = Other.hasBoundingBox;
		hasBones = Other.hasBones;
	}

	void MeshData::ComputeBounding() {
		if (!hasBoundingBox) {
			Bounding = BoundingBox3D();
			for (MeshVertices::const_iterator Vertex = StaticVertices.begin(); Vertex != StaticVertices.end(); ++Vertex) {
				Bounding.Add(Vertex->Position);
			}
			hasBoundingBox = true;
		}
	}

	void MeshData::ComputeTangents() {
		if (UVChannels <= 0 || hasTangents)
			return;

		// --- For each triangle, compute the edge (DeltaPos) and the DeltaUV
		for (MeshFaces::const_iterator Triangle = Faces.begin(); Triangle != Faces.end(); ++Triangle) {
			const Vector3 & VertexA = StaticVertices[(*Triangle)[0]].Position;
			const Vector3 & VertexB = StaticVertices[(*Triangle)[1]].Position;
			const Vector3 & VertexC = StaticVertices[(*Triangle)[2]].Position;

			const Vector2 & UVA = StaticVertices[(*Triangle)[0]].UV0;
			const Vector2 & UVB = StaticVertices[(*Triangle)[1]].UV0;
			const Vector2 & UVC = StaticVertices[(*Triangle)[2]].UV0;

			// --- Edges of the triangle : position delta
			const Vector3 Edge1 = VertexB - VertexA;
			const Vector3 Edge2 = VertexC - VertexA;

			// --- UV delta
			const Vector2 DeltaUV1 = UVB - UVA;
			const Vector2 DeltaUV2 = UVC - UVA;

			float r = 1.F / (DeltaUV1.x * DeltaUV2.y - DeltaUV1.y * DeltaUV2.x);
			r = std::isfinite(r) ? r : 0;

			Vector3 Tangent;
			Tangent.x = r * (DeltaUV2.y * Edge1.x - DeltaUV1.y * Edge2.x);
			Tangent.y = r * (DeltaUV2.y * Edge1.y - DeltaUV1.y * Edge2.y);
			Tangent.z = r * (DeltaUV2.y * Edge1.z - DeltaUV1.y * Edge2.z);
			Tangent.Normalize();

			StaticVertices[(*Triangle)[0]].Tangent = Tangent;
			StaticVertices[(*Triangle)[1]].Tangent = Tangent;
			StaticVertices[(*Triangle)[2]].Tangent = Tangent;
		}

		hasTangents = true;
	}

	void MeshData::ComputeNormals() {
		if (StaticVertices.size() <= 0 || hasNormals)
			return;

		for (MeshFaces::const_iterator Triangle = Faces.begin(); Triangle != Faces.end(); ++Triangle) {
			const Vector3 & VertexA = StaticVertices[(*Triangle)[0]].Position;
			const Vector3 & VertexB = StaticVertices[(*Triangle)[1]].Position;
			const Vector3 & VertexC = StaticVertices[(*Triangle)[2]].Position;

			const Vector3 Edge1 = VertexB - VertexA;
			const Vector3 Edge2 = VertexC - VertexA;

			const Vector3 Normal = Vector3::Cross(Edge1, Edge2).Normalized();

			StaticVertices[(*Triangle)[0]].Normal = Normal;
			StaticVertices[(*Triangle)[1]].Normal = Normal;
			StaticVertices[(*Triangle)[2]].Normal = Normal;
		}

		hasNormals = true;
	}

	void MeshData::Clear() {
		Name.clear();
		Faces.clear();
		StaticVertices.clear();
		Bounding = BoundingBox3D();
		hasNormals = false;
		hasTangents = false;
		hasVertexColor = false;
		UVChannels = 0;
		hasBoundingBox = true;
		hasBones = false;
	}

}