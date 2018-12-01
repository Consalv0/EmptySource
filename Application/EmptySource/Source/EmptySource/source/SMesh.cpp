
#include "..\include\SCore.h"

#include "..\include\SMath.h"
#include "..\include\SMesh.h"

SMesh::SMesh() {
	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	Triangles = SMeshTriangles();
	Vertices = SMeshVertices();
}

SMesh::SMesh( 
	const SMeshTriangles triangles, const SMeshVector3D vertices,
	const SMeshVector3D normals, const SMeshUVs uv0, const SMeshColors colors)
{
	Triangles = triangles;
	for (int vCount = 0; vCount < vertices.size(); vCount++) {
		Vertices.push_back(Vertex({ vertices[vCount], normals[vCount], FVector3(), uv0[vCount], FVector2(), colors[vCount] }));
	}

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

SMesh::SMesh(
	const SMeshTriangles triangles, const SMeshVector3D vertices,
	const SMeshVector3D normals, const SMeshVector3D tangents, const SMeshUVs uv0, const SMeshColors colors)
{
	Triangles = triangles;
	for (int vCount = 0; vCount < vertices.size(); vCount++) {
		Vertices.push_back(Vertex({ vertices[vCount], normals[vCount], tangents[vCount], uv0[vCount], FVector2(), colors[vCount] }));
	}

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

SMesh SMesh::BuildCube() {
	SMesh TemporalMesh;
	static const SMeshTriangles TemporalTriangles{
		// Front Face
		{  0,  1,  2 }, {  2,  3,  0 },
		// Back Face
		{  4,  5,  6 }, {  6,  7,  4 },
		// Right Face
		{  8,  9, 10 }, { 10, 11,  8 },
		// Left Face
		{ 12, 13, 14 }, { 14, 15, 12 },
		// Up Face
		{ 16, 17, 18 }, { 18, 19, 16 },
		// Down Face
		{ 20, 21, 22 }, { 22, 23, 20 },
	};
	static const SMeshVector3D TemporalVertices{
		// Front Face
		{ 0.5F, -0.5F, -0.5F }, // 1 : 1
		{-0.5F, -0.5F, -0.5F }, // 2 : 2
		{-0.5F,  0.5F, -0.5F }, // 6 : 3
		{ 0.5F,  0.5F, -0.5F }, // 3 : 5

		// Back Face
		{ -0.5F,  0.5F,  0.5F }, // 5 : 7
		{  0.5F,  0.5F,  0.5F }, // 4 : 8
		{  0.5F, -0.5F,  0.5F }, // 8 : 10
		{ -0.5F, -0.5F,  0.5F }, // 7 : 11

		// Right Face
		{ 0.5F, -0.5F, -0.5F }, // 1 : 13
		{ 0.5F,  0.5F, -0.5F }, // 3 : 14
		{ 0.5F,  0.5F,  0.5F }, // 4 : 16
		{ 0.5F, -0.5F,  0.5F }, // 8 : 17

		// Left Face
		{-0.5F,  0.5F,  0.5F }, // 5 : 19
		{-0.5F, -0.5F,  0.5F }, // 7 : 20
		{-0.5F, -0.5F, -0.5F }, // 2 : 22
		{-0.5F,  0.5F, -0.5F }, // 6 : 23

		// Up Face
		{-0.5F,  0.5F,  0.5F }, // 5 : 25
		{ 0.5F,  0.5F,  0.5F }, // 4 : 26
		{ 0.5F,  0.5F, -0.5F }, // 3 : 28
		{-0.5F,  0.5F, -0.5F }, // 6 : 29

		// Down Face
		{ 0.5F, -0.5F, -0.5F }, // 1 : 31
		{-0.5F, -0.5F, -0.5F }, // 2 : 32
		{-0.5F, -0.5F,  0.5F }, // 7 : 34
		{ 0.5F, -0.5F,  0.5F }, // 8 : 35
	};
	static const SMeshVector3D TemporalNormals{
		// Front Face
		{ 0.0F,  0.0F, -1.0F }, // 1
		{ 0.0F,  0.0F, -1.0F }, // 2
		{ 0.0F,  0.0F, -1.0F }, // 6
		{ 0.0F,  0.0F, -1.0F }, // 3

		// Back Face		 
		{ 0.0F,  0.0F,  1.0F }, // 5
		{ 0.0F,  0.0F,  1.0F }, // 4
		{ 0.0F,  0.0F,  1.0F }, // 8
		{ 0.0F,  0.0F,  1.0F }, // 7

		// Right Face		 
		{ 1.0F,  0.0F,  0.0F }, // 1
		{ 1.0F,  0.0F,  0.0F }, // 3
		{ 1.0F,  0.0F,  0.0F }, // 4
		{ 1.0F,  0.0F,  0.0F }, // 8

		// Left Face		 
		{-1.0F,  0.0F,  0.0F }, // 5
		{-1.0F,  0.0F,  0.0F }, // 7
		{-1.0F,  0.0F,  0.0F }, // 2
		{-1.0F,  0.0F,  0.0F }, // 6

		// Up Face			 
		{ 0.0F,  1.0F,  0.0F }, // 5
		{ 0.0F,  1.0F,  0.0F }, // 4
		{ 0.0F,  1.0F,  0.0F }, // 3
		{ 0.0F,  1.0F,  0.0F }, // 6

		// Down Face		 
		{ 0.0F, -1.0F,  0.0F }, // 2
		{ 0.0F, -1.0F,  0.0F }, // 1
		{ 0.0F, -1.0F,  0.0F }, // 7
		{ 0.0F, -1.0F,  0.0F }, // 8
	};
	static const SMeshUVs      TemporalTextureCoords{
		// Front Face
		{ 1.0F, -1.0F }, // 1
		{-1.0F, -1.0F }, // 2
		{-1.0F,  1.0F }, // 6
		{ 1.0F,  1.0F }, // 3

		// Back Face
		{-1.0F,  1.0F }, // 5
		{ 1.0F,  1.0F }, // 4
		{ 1.0F, -1.0F }, // 8
		{-1.0F, -1.0F }, // 7

		// Right Face
		{ 1.0F, -1.0F }, // 1
		{ 1.0F,  1.0F }, // 3
		{ 1.0F,  1.0F }, // 4
		{ 1.0F, -1.0F }, // 8

		// Left Face
		{-1.0F,  1.0F }, // 5
		{-1.0F, -1.0F }, // 7
		{-1.0F, -1.0F }, // 2
		{-1.0F,  1.0F }, // 6

		// Up Face
		{-1.0F,  1.0F }, // 5
		{ 1.0F,  1.0F }, // 4
		{ 1.0F,  1.0F }, // 3
		{-1.0F,  1.0F }, // 6

		// Down Face
		{ 1.0F, -1.0F }, // 1
		{-1.0F, -1.0F }, // 2
		{-1.0F, -1.0F }, // 7
		{ 1.0F, -1.0F }, // 8
	};
	static const SMeshColors   TemporalColors{
		// Front Face
		{ 0.0F, 0.0F, 1.0F, 1.0F },
		{ 0.0F, 0.0F, 1.0F, 1.0F },
		{ 0.0F, 0.0F, 1.0F, 1.0F },
		{ 0.0F, 0.0F, 1.0F, 1.0F },

		// Back Face
		{ 0.0F, 1.0F, 0.0F, 1.0F },
		{ 0.0F, 1.0F, 0.0F, 1.0F },
		{ 0.0F, 1.0F, 0.0F, 1.0F },
		{ 0.0F, 1.0F, 0.0F, 1.0F },

		// Right Face
		{ 1.0F, 0.0F, 0.0F, 1.0F },
		{ 1.0F, 0.0F, 0.0F, 1.0F },
		{ 1.0F, 0.0F, 0.0F, 1.0F },
		{ 1.0F, 0.0F, 0.0F, 1.0F },

		// Left Face
		{ 0.0F, 1.0F, 1.0F, 1.0F },
		{ 0.0F, 1.0F, 1.0F, 1.0F },
		{ 0.0F, 1.0F, 1.0F, 1.0F },
		{ 0.0F, 1.0F, 1.0F, 1.0F },

		// Up Face
		{ 1.0F, 1.0F, 0.0F, 1.0F },
		{ 1.0F, 1.0F, 0.0F, 1.0F },
		{ 1.0F, 1.0F, 0.0F, 1.0F },
		{ 1.0F, 1.0F, 0.0F, 1.0F },

		// Down Face
		{ 1.0F, 1.0F, 1.0F, 1.0F },
		{ 1.0F, 1.0F, 1.0F, 1.0F },
		{ 1.0F, 1.0F, 1.0F, 1.0F },
		{ 1.0F, 1.0F, 1.0F, 1.0F },
	};

	TemporalMesh = SMesh(TemporalTriangles, TemporalVertices, TemporalNormals, TemporalTextureCoords, TemporalColors);

	return TemporalMesh;
}

void SMesh::BindVertexArray() {
	// Generate 1 VAO, put the resulting identifier in VAO identifier
	if (VertexArrayObject == 0) glGenVertexArrays(1, &VertexArrayObject);
	// The following commands will put in context our VAO for the next commands
	glBindVertexArray(VertexArrayObject);
}

void SMesh::DrawInstanciated(int Count) const {
	glDrawElementsInstanced(
		GL_TRIANGLES,	                        // mode
		(int)Triangles.size() * 3,	            // mode count
		GL_UNSIGNED_INT,                        // type
		(void*)0,		                        // element array buffer offset
		Count                                   // element count
	);
}

void SMesh::DrawElement() const {
	glDrawElements(
		GL_TRIANGLES,	                        // mode
		(int)Triangles.size() * 3,	            // mode count
		GL_UNSIGNED_INT,                        // type
		(void*)0		                        // element array buffer offset
	); // Starting from vertex 0; to vertices total
}

void SMesh::SetUpBuffers() {

	BindVertexArray();

	// This will identify our vertex buffer
	glGenBuffers(1, &VertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, Vertices.size() * sizeof(Vertex), &Vertices[0], GL_STATIC_DRAW);

	// Generate a Element Buffer for the indices
	glGenBuffers(1, &ElementBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElementBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, Triangles.size() * sizeof(IVector3), &Triangles[0], GL_STATIC_DRAW);

	// Set the vertex attribute pointers
	// Positions
	glEnableVertexAttribArray(VertexLocation);
	glVertexAttribPointer(VertexLocation, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	// Normals
	glEnableVertexAttribArray(NormalLocation);
	glVertexAttribPointer(NormalLocation, 3, GL_FLOAT, GL_TRUE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
	// Tangent
	glEnableVertexAttribArray(TangentLocation);
	glVertexAttribPointer(TangentLocation, 3, GL_FLOAT, GL_TRUE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
	// Texture Coords
	glEnableVertexAttribArray(UV0Location);
	glVertexAttribPointer(UV0Location, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, UV0));
	glEnableVertexAttribArray(UV1Location);
	glVertexAttribPointer(UV1Location, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, UV1));
	// Color
	glEnableVertexAttribArray(ColorLocation);
	glVertexAttribPointer(ColorLocation, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Color));

	glBindVertexArray(0);
}

void SMesh::ClearBuffers() {
	glDeleteVertexArrays(1, &VertexArrayObject);
}
