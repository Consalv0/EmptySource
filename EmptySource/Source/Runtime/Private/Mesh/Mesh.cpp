
#include "Engine/Log.h"
#include "Engine/Core.h"
#include "Graphics/GLFunctions.h"
#include "Mesh/Mesh.h"

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
		ElementBufferObject = ElementBuffer();
		VertexBuffer = 0;
		ElementBufferSubdivisions = TArray<ElementBuffer>();
		Data = MeshData();
	}

	Mesh::Mesh(const MeshData & OtherData) {
		Data = OtherData;

		ElementBufferObject = ElementBuffer();
		VertexBuffer = 0;
		ElementBufferSubdivisions = TArray<ElementBuffer>();
	}

	Mesh::Mesh(MeshData * OtherData) {
		Data.Swap(*OtherData);

		ElementBufferObject = ElementBuffer();
		VertexBuffer = 0;
		ElementBufferSubdivisions = TArray<ElementBuffer>();
	}

	void Mesh::BindVertexArray() const {
		// Generate 1 VAO, put the resulting identifier in VAO identifier
		if (!ElementBufferObject.IsValid()) {
			LOG_CORE_WARN(L"Model buffers are empty, use SetUpBuffers first");
			return;
		}

		ElementBufferObject.Bind();
	}

	void Mesh::BindSubdivisionVertexArray(int MaterialIndex) const {
		auto Subdivision = Data.MaterialSubdivisions.find(MaterialIndex);
		if (Subdivision == Data.MaterialSubdivisions.end()) return;
		if (MaterialIndex >= ElementBufferSubdivisions.size() ||
			!ElementBufferSubdivisions[MaterialIndex].IsValid()) return;

		ElementBufferSubdivisions[MaterialIndex].Bind();

		return;
	}

	void Mesh::DrawInstanciated(int Count) const {
		if (!ElementBufferObject.IsValid()) return;

		glDrawElementsInstanced(
			GL_TRIANGLES,	             // mode
			(int)Data.Faces.size() * 3,	 // mode count
			GL_UNSIGNED_INT,             // type
			(void*)0,		             // element array buffer offset
			Count                        // element count
		);
	}

	void Mesh::DrawSubdivisionInstanciated(int Count, int MaterialIndex) const {
		auto Subdivision = Data.MaterialSubdivisions.find(MaterialIndex);
		if (Subdivision == Data.MaterialSubdivisions.end()) return;
		if (MaterialIndex >= ElementBufferSubdivisions.size() ||
			!ElementBufferSubdivisions[MaterialIndex].IsValid()) return;

		glDrawElementsInstanced(
			GL_TRIANGLES,	             // mode
			(int)Subdivision->second.size() * 3,	// mode count
			GL_UNSIGNED_INT,             // type
			(void*)0,		             // element array buffer offset
			Count                        // element count
		);
	}

	void Mesh::DrawElement() const {
		if (!ElementBufferObject.IsValid()) return;

		glDrawElements(
			GL_TRIANGLES,	             // mode
			(int)Data.Faces.size() * 3,	 // mode count
			GL_UNSIGNED_INT,             // type
			(void*)0		             // element array buffer offset
		); // Starting from vertex 0; to vertices total
	}

	bool Mesh::SetUpBuffers() {

		if (Data.Vertices.size() <= 0 || Data.Faces.size() <= 0) return false;
		if (VertexBuffer != 0 && ElementBufferObject.IsValid()) return true;

		// This will identify our vertex buffer
		glGenBuffers(1, &VertexBuffer);
		// Give our vertices to OpenGL.
		glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
		glBufferData(GL_ARRAY_BUFFER, Data.Vertices.size() * sizeof(MeshVertex), &Data.Vertices[0], GL_STATIC_DRAW);

		for (int ElementBufferCount = 0; ElementBufferCount <= Data.MaterialSubdivisions.size(); ElementBufferCount++) {
			// Generate a Element Buffer from indices
			if (ElementBufferCount == 0)
				ElementBufferObject.SetUpBuffers(VertexBuffer, Data.Faces);
			else {
				ElementBufferSubdivisions.push_back(ElementBuffer());
				ElementBufferSubdivisions.back().SetUpBuffers(VertexBuffer, Data.MaterialSubdivisions[ElementBufferCount - 1]);
			}

			glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);

			// Set the vertex attribute pointers layouts
			glEnableVertexAttribArray(PositionLocation);
			glVertexAttribPointer(PositionLocation, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)0);
			glEnableVertexAttribArray(NormalLocation);
			glVertexAttribPointer(NormalLocation, 3, GL_FLOAT, GL_TRUE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Normal));
			glEnableVertexAttribArray(TangentLocation);
			glVertexAttribPointer(TangentLocation, 3, GL_FLOAT, GL_TRUE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Tangent));
			glEnableVertexAttribArray(UV0Location);
			glVertexAttribPointer(UV0Location, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, UV0));
			glEnableVertexAttribArray(UV1Location);
			glVertexAttribPointer(UV1Location, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, UV1));
			glEnableVertexAttribArray(ColorLocation);
			glVertexAttribPointer(ColorLocation, 4, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Color));
		}

		glEnableVertexAttribArray(0);
		glBindVertexArray(0);

		return true;
	}

	void Mesh::ClearBuffers() {
		glDeleteBuffers(1, &VertexBuffer);

		ElementBufferObject.Clear();
		for (ElementBuffer& EBO : ElementBufferSubdivisions) {
			EBO.Clear();
		}
		ElementBufferSubdivisions.clear();
		VertexBuffer = 0;
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

	void Mesh::ElementBuffer::Clear() {
		glDeleteVertexArrays(1, &VertexArrayObject);
		glDeleteBuffers(1, &Buffer);

		VertexArrayObject = 0;
		Buffer = 0;
	}

	bool Mesh::ElementBuffer::SetUpBuffers(const unsigned int & VertexBuffer, const MeshFaces & Indices) {
		if (Indices.empty()) return false;
		if (VertexArrayObject != 0 && Buffer != 0) return true;

		glGenVertexArrays(1, &VertexArrayObject);
		glBindVertexArray(VertexArrayObject);

		// Generate a Element Buffer for the indices
		glGenBuffers(1, &Buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, Indices.size() * sizeof(IntVector3), &Indices[0], GL_STATIC_DRAW);

		return true;
	}

	void Mesh::ElementBuffer::Bind() const {
		// The following commands will put in context our VAO for the next commands
		glBindVertexArray(VertexArrayObject);
	}

	bool Mesh::ElementBuffer::IsValid() const {
		return VertexArrayObject != 0 && Buffer != 0;
	}

}