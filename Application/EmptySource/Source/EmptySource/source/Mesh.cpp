
#include "..\include\Graphics.h"
#include "..\include\Core.h"

#include "..\include\Math\Math.h"
#include "..\include\Mesh.h"

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

Mesh::Mesh( 
	const MeshFaces faces, const MeshVector3D vertices,
	const MeshVector3D normals, const MeshUVs uv0, const MeshColors colors)
{
	Faces = faces;
	for (int vCount = 0; vCount < vertices.size(); vCount++) {
		Vertices.push_back(MeshVertex({ vertices[vCount], normals[vCount], Vector3(), uv0[vCount], Vector2(), colors[vCount] }));
	}

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

Mesh::Mesh(const MeshFaces faces, const MeshVector3D vertices, const MeshVector3D normals, const MeshUVs uv0, const MeshUVs uv1, const MeshColors colors) {
	Faces = faces;
	for (int vCount = 0; vCount < vertices.size(); vCount++) {
		Vertices.push_back(MeshVertex({ vertices[vCount], normals[vCount], Vector3(), uv0[vCount],  uv1[vCount], colors[vCount] }));
	}

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

Mesh::Mesh(
	const MeshFaces faces, const MeshVector3D vertices,
	const MeshVector3D normals, const MeshVector3D tangents, 
	const MeshUVs uv0, const MeshColors colors) 

	: Mesh(faces, vertices, normals, tangents, uv0, uv0, colors)
{
}

Mesh::Mesh(
	const MeshFaces faces, const MeshVector3D vertices,
	const MeshVector3D normals, const MeshVector3D tangents,
	const MeshUVs uv0, const MeshUVs uv1,
	const MeshColors colors)
{
	Faces = faces;
	for (int vCount = 0; vCount < vertices.size(); vCount++) {
		Vertices.push_back(MeshVertex({ vertices[vCount], normals[vCount], tangents[vCount], uv0[vCount], Vector2(), colors[vCount] }));
	}

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

Mesh::Mesh(const MeshFaces faces, const MeshVertices vertices) {
	Faces = faces;
	Vertices = vertices;

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

Mesh::Mesh(MeshFaces * faces, MeshVertices * vertices) {
	Faces.swap(*faces);
	Vertices.swap(*vertices);

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

void Mesh::BindVertexArray() const {
	// Generate 1 VAO, put the resulting identifier in VAO identifier
	if (VertexArrayObject == 0) {
		Debug::Log(Debug::LogWarning, L"Model buffers are empty, use SetUpBuffers first");
		return;
	}
	// The following commands will put in context our VAO for the next commands
	glBindVertexArray(VertexArrayObject);
}

void Mesh::DrawInstanciated(int Count) const {
	BindVertexArray();

	glDrawElementsInstanced(
		GL_TRIANGLES,	                        // mode
		(int)Faces.size() * 3,	                // mode count
		GL_UNSIGNED_INT,                        // type
		(void*)0,		                        // element array buffer offset
		Count                                   // element count
	);
}

void Mesh::DrawElement() const {
	glDrawElements(
		GL_TRIANGLES,	                        // mode
		(int)Faces.size() * 3,	                // mode count
		GL_UNSIGNED_INT,                        // type
		(void*)0		                        // element array buffer offset
	); // Starting from vertex 0; to vertices total
}

void Mesh::SetUpBuffers() {

	if (Vertices.size() <= 0 || Faces.size() <= 0) return;

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
	glEnableVertexAttribArray(  VertexLocation );
	    glVertexAttribPointer(  VertexLocation , 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)0);
	glEnableVertexAttribArray(  NormalLocation );
	    glVertexAttribPointer(  NormalLocation , 3, GL_FLOAT,  GL_TRUE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Normal));
	glEnableVertexAttribArray( TangentLocation );
	    glVertexAttribPointer( TangentLocation , 3, GL_FLOAT,  GL_TRUE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Tangent));
	glEnableVertexAttribArray(     UV0Location );
	    glVertexAttribPointer(     UV0Location , 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, UV0));
	glEnableVertexAttribArray(     UV1Location );
	    glVertexAttribPointer(     UV1Location , 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, UV1));
	glEnableVertexAttribArray(   ColorLocation );
	    glVertexAttribPointer(   ColorLocation , 4, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (void*)offsetof(MeshVertex, Color));

	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}

void Mesh::ClearBuffers() {
	glDeleteVertexArrays(1, &VertexArrayObject);
}
