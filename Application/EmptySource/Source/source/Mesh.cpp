
#include "../include/Core.h"
#include "../include/CoreGraphics.h"

#include "../include/Math/CoreMath.h"
#include "../include/Math/Box3D.h"
#include "../include/Mesh.h"

MeshVertex::MeshVertex(const Vector3 & P, const Vector3 & N, const Vector2 & UV) :
	Position(P), Normal(N), Tangent(), UV0(UV), UV1(UV), Color() {
}

MeshVertex::MeshVertex(const Vector3 & P, const Vector3 & N, const Vector3 & T, const Vector2 & UV0, const Vector2 & UV1, const Vector4 & C) :
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
	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	Faces = MeshFaces();
	Vertices = MeshVertices();
}

Mesh::Mesh(const MeshFaces faces, const MeshVertices vertices) {
	Faces = faces;
	Vertices = vertices;

	Bounding = BoundingBox3D();
	for (MeshVertices::const_iterator Vertex = Vertices.begin(); Vertex != Vertices.end(); ++Vertex) {
		Bounding.Add(Vertex->Position);
	}
	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
}

Mesh::Mesh(MeshFaces * faces, MeshVertices * vertices) {
	Faces.swap(*faces);
	Vertices.swap(*vertices);

	Bounding = BoundingBox3D();
	for (MeshVertices::const_iterator Vertex = Vertices.begin(); Vertex != Vertices.end(); ++Vertex) {
		Bounding.Add(Vertex->Position);
	}
	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
}

Mesh::Mesh(MeshFaces * faces, MeshVertices * vertices, const Box3D & BBox) {
	Faces.swap(*faces);
	Vertices.swap(*vertices);

	Bounding = BBox;
	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
}

void Mesh::BindVertexArray() const {
	// Generate 1 VAO, put the resulting identifier in VAO identifier
	if (VertexArrayObject == 0) {
		Debug::Log(Debug::LogWarning, L"Model buffers are empty, use SetUpBuffers first");
		return;
	}
	// The following commands will put in context our VAO for the next commands
	glBindVertexArray(VertexArrayObject);
	return;
}

void Mesh::DrawInstanciated(int Count) const {
	if (VertexArrayObject == 0) return;

	glDrawElementsInstanced(
		GL_TRIANGLES,	         // mode
		(int)Faces.size() * 3,	 // mode count
		GL_UNSIGNED_INT,         // type
		(void*)0,		         // element array buffer offset
		Count                    // element count
	);
}

void Mesh::DrawElement() const {
	if (VertexArrayObject == 0) return;

	glDrawElements(
		GL_TRIANGLES,	                        // mode
		(int)Faces.size() * 3,	                // mode count
		GL_UNSIGNED_INT,                        // type
		(void*)0		                        // element array buffer offset
	); // Starting from vertex 0; to vertices total
}

bool Mesh::SetUpBuffers() {

	if (Vertices.size() <= 0 || Faces.size() <= 0) return false;
	if (VertexArrayObject != 0 && VertexBuffer != 0 && ElementBuffer != 0) return true;

	glGenVertexArrays(1, &VertexArrayObject);
	glBindVertexArray(VertexArrayObject);

	// This will identify our vertex buffer
	glGenBuffers(1, &VertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, Vertices.size() * sizeof(MeshVertex), &Vertices[0], GL_STATIC_DRAW);

	// Generate a Element Buffer for the indices
	glGenBuffers(1, &ElementBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElementBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, Faces.size() * sizeof(IntVector3), &Faces[0], GL_STATIC_DRAW);

	// Set the vertex attribute pointers layouts
	glEnableVertexAttribArray( PositionLocation );
	    glVertexAttribPointer( PositionLocation , 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)0);
	glEnableVertexAttribArray(   NormalLocation );
	    glVertexAttribPointer(   NormalLocation , 3, GL_FLOAT,  GL_TRUE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Normal));
	glEnableVertexAttribArray(  TangentLocation );
	    glVertexAttribPointer(  TangentLocation , 3, GL_FLOAT,  GL_TRUE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Tangent));
	glEnableVertexAttribArray(      UV0Location );
	    glVertexAttribPointer(      UV0Location , 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, UV0));
	glEnableVertexAttribArray(      UV1Location );
	    glVertexAttribPointer(      UV1Location , 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, UV1));
	glEnableVertexAttribArray(    ColorLocation );
	    glVertexAttribPointer(    ColorLocation , 4, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Color));

	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	return true;
}

void Mesh::ClearBuffers() {
	glDeleteVertexArrays(1, &VertexArrayObject);
	glDeleteBuffers(1, &VertexBuffer);
	glDeleteBuffers(1, &ElementBuffer);

	VertexArrayObject = 0;
	VertexBuffer = 0;
	ElementBuffer = 0;
}

void Mesh::Clear() {
	Faces.clear();
	Vertices.clear();
	ClearBuffers();
}
