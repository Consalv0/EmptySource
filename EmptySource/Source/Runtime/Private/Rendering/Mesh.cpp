
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/Mesh.h"

namespace EmptySource {

	MeshVertex::MeshVertex(const Vector3 & P, const Vector3 & N, const Vector2 & UV) :
		Position(P), Normal(N), Tangent(), UV0(UV), UV1(UV), Color() {
	}

	MeshVertex::MeshVertex(const Vector3 & P, const Vector3 & N, const Vector3 & T,
		const Vector2 & UV0, const Vector2 & UV1, const Vector4 & C) :
		Position(P), Normal(N), Tangent(T),
		UV0(UV0), UV1(UV1), Color(C) {
	}

	bool MeshVertex::operator<(const MeshVertex that) const {
		return memcmp((void*)this, (void*)&that, sizeof(MeshVertex)) > 0;
	}

	bool MeshVertex::operator==(const MeshVertex & other) const {
		return (Position == other.Position
			&& Normal == other.Normal
			&& Tangent == other.Tangent
			&& UV0 == other.UV0
			&& UV1 == other.UV1
			&& Color == other.Color);
	};

	Mesh::Mesh() {
		MeshSubdivisions = TArray<VertexArrayPtr>(NULL);
		Data = MeshData();
	}

	Mesh::Mesh(const MeshData & OtherData) {
		MeshSubdivisions = TArray<VertexArrayPtr>(NULL);
		Data = OtherData;
	}

	Mesh::Mesh(MeshData * OtherData) {
		MeshSubdivisions = TArray<VertexArrayPtr>(NULL);
		Data.Swap(*OtherData);
	}

	void Mesh::SwapMeshData(MeshData & NewData) {
		Clear(); Data.Swap(NewData);
	}

	void Mesh::CopyMeshData(const MeshData & NewData) {
		Clear(); Data = NewData;
	}

	void Mesh::BindSubdivisionVertexArray(int MaterialIndex) const {
		auto Subdivision = Data.MaterialSubdivisions.find(MaterialIndex);
		if (Subdivision == Data.MaterialSubdivisions.end()) return;
		if (MaterialIndex >= MeshSubdivisions.size() || MeshSubdivisions[MaterialIndex] == NULL) return;

		MeshSubdivisions[MaterialIndex]->Bind();

		return;
	}

	VertexArrayPtr Mesh::GetSubdivisionVertexArray(int MaterialIndex) const {
		auto Subdivision = Data.MaterialSubdivisions.find(MaterialIndex);
		if (Subdivision == Data.MaterialSubdivisions.end()) return NULL;
		if (MaterialIndex >= MeshSubdivisions.size() || MeshSubdivisions[MaterialIndex] == NULL) return NULL;

		return MeshSubdivisions[MaterialIndex];
	}

	void Mesh::DrawSubdivisionInstanciated(int Count, int MaterialIndex) const {
		auto Subdivision = Data.MaterialSubdivisions.find(MaterialIndex);
		if (Subdivision == Data.MaterialSubdivisions.end()) return;
		if (MaterialIndex >= MeshSubdivisions.size() || MeshSubdivisions[MaterialIndex] == NULL) return;

		Rendering::DrawIndexed(MeshSubdivisions[MaterialIndex], Count);
	}

	bool Mesh::SetUpBuffers() {

		if (Data.Vertices.size() <= 0 || Data.Faces.size() <= 0) return false;

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
		VertexBufferPointer = VertexBuffer::Create((float *)&Data.Vertices[0], (unsigned int)(Data.Vertices.size() * sizeof(MeshVertex)), UM_Static);
		VertexBufferPointer->SetLayout(DafultLayout);

		if (Data.MaterialSubdivisions.size() <= 0) {
			Data.Materials.emplace(0, "default");
			Data.MaterialSubdivisions.emplace(0, Data.Faces);
		}

		for (int ElementBufferCount = 0; ElementBufferCount < Data.MaterialSubdivisions.size(); ElementBufferCount++) {
			VertexArrayPtr VertexArrayPointer = NULL;
			VertexArrayPointer = VertexArray::Create();
			IndexBufferPtr IndexBufferPointer = NULL;
			IndexBufferPointer = IndexBuffer::Create(
				(unsigned int *)&Data.MaterialSubdivisions[ElementBufferCount][0],
				(unsigned int)Data.MaterialSubdivisions[ElementBufferCount].size() * 3, UM_Static
			);
			VertexArrayPointer->AddVertexBuffer(VertexBufferPointer);
			VertexArrayPointer->AddIndexBuffer(IndexBufferPointer);
			VertexArrayPointer->Unbind();
			
			MeshSubdivisions.push_back(VertexArrayPointer);
		}

		return true;
	}

	void Mesh::ClearBuffers() {
		MeshSubdivisions.clear();
	}

	void Mesh::Clear() {
		Data.Clear();
		ClearBuffers();
	}

	void MeshData::Swap(MeshData & Other) {
		Name = Other.Name;
		Faces.swap(Other.Faces);
		Vertices.swap(Other.Vertices);
		MaterialSubdivisions.swap(Other.MaterialSubdivisions);
		Materials.swap(Other.Materials);
		Bounding = Other.Bounding;

		hasNormals = Other.hasNormals;
		hasTangents = Other.hasTangents;
		hasVertexColor = Other.hasVertexColor;
		TextureCoordsCount = Other.TextureCoordsCount;
		hasBoundingBox = Other.hasBoundingBox;
		hasWeights = Other.hasWeights;
	}

	void MeshData::ComputeBounding() {
		if (!hasBoundingBox) {
			Bounding = BoundingBox3D();
			for (MeshVertices::const_iterator Vertex = Vertices.begin(); Vertex != Vertices.end(); ++Vertex) {
				Bounding.Add(Vertex->Position);
			}
			hasBoundingBox = true;
		}
	}

	void MeshData::ComputeTangents() {
		if (TextureCoordsCount <= 0 || hasTangents)
			return;

		// --- For each triangle, compute the edge (DeltaPos) and the DeltaUV
		for (MeshFaces::const_iterator Triangle = Faces.begin(); Triangle != Faces.end(); ++Triangle) {
			const Vector3 & VertexA = Vertices[(*Triangle)[0]].Position;
			const Vector3 & VertexB = Vertices[(*Triangle)[1]].Position;
			const Vector3 & VertexC = Vertices[(*Triangle)[2]].Position;

			const Vector2 & UVA = Vertices[(*Triangle)[0]].UV0;
			const Vector2 & UVB = Vertices[(*Triangle)[1]].UV0;
			const Vector2 & UVC = Vertices[(*Triangle)[2]].UV0;

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

			Vertices[(*Triangle)[0]].Tangent = Tangent;
			Vertices[(*Triangle)[1]].Tangent = Tangent;
			Vertices[(*Triangle)[2]].Tangent = Tangent;
		}

		hasTangents = true;
	}

	void MeshData::ComputeNormals() {
		if (Vertices.size() <= 0 || hasNormals)
			return;

		for (MeshFaces::const_iterator Triangle = Faces.begin(); Triangle != Faces.end(); ++Triangle) {
			const Vector3 & VertexA = Vertices[(*Triangle)[0]].Position;
			const Vector3 & VertexB = Vertices[(*Triangle)[1]].Position;
			const Vector3 & VertexC = Vertices[(*Triangle)[2]].Position;

			const Vector3 Edge1 = VertexB - VertexA;
			const Vector3 Edge2 = VertexC - VertexA;

			const Vector3 Normal = Vector3::Cross(Edge1, Edge2).Normalized();

			Vertices[(*Triangle)[0]].Normal = Normal;
			Vertices[(*Triangle)[1]].Normal = Normal;
			Vertices[(*Triangle)[2]].Normal = Normal;
		}

		hasNormals = true;
	}

	void MeshData::Clear() {
		Name.clear();
		Faces.clear();
		Vertices.clear();
		Bounding = BoundingBox3D();
		hasNormals = false;
		hasTangents = false;
		hasVertexColor = false;
		TextureCoordsCount = 0;
		hasBoundingBox = true;
		hasWeights = false;
	}

}