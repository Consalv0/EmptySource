
#include "..\include\Core.h"

#include "..\include\Math\Math.h"
#include "..\include\Mesh.h"

Mesh::Mesh() {
	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	Triangles = MeshTriangles();
	Vertices = MeshVertices();
}

Mesh::Mesh( 
	const MeshTriangles triangles, const MeshVector3D vertices,
	const MeshVector3D normals, const MeshUVs uv0, const MeshColors colors)
{
	Triangles = triangles;
	for (int vCount = 0; vCount < vertices.size(); vCount++) {
		Vertices.push_back(Vertex({ vertices[vCount], normals[vCount], Vector3(), uv0[vCount], Vector2(), colors[vCount] }));
	}

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

Mesh::Mesh(
	const MeshTriangles triangles, const MeshVector3D vertices,
	const MeshVector3D normals, const MeshVector3D tangents, const MeshUVs uv0, const MeshColors colors) 
	
	: Mesh(triangles, vertices, normals, tangents, uv0, uv0, colors)
{
}

Mesh::Mesh(
	const MeshTriangles triangles, const MeshVector3D vertices,
	const MeshVector3D normals, const MeshVector3D tangents,
	const MeshUVs uv0, const MeshUVs uv1,
	const MeshColors colors)
{
	Triangles = triangles;
	for (int vCount = 0; vCount < vertices.size(); vCount++) {
		Vertices.push_back(Vertex({ vertices[vCount], normals[vCount], tangents[vCount], uv0[vCount], Vector2(), colors[vCount] }));
	}

	VertexArrayObject = 0;
	ElementBuffer = 0;
	VertexBuffer = 0;
	SetUpBuffers();
}

Mesh Mesh::BuildCube() {
	Mesh TemporalMesh;
	static const MeshTriangles TemporalTriangles{
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
	static const MeshVector3D TemporalVertices{
		// Front Face
		{ 0.5F, -0.5F, -0.5F }, // 1 : 1
		{-0.5F, -0.5F, -0.5F }, // 2 : 2
		{-0.5F,  0.5F, -0.5F }, // 6 : 3
		{ 0.5F,  0.5F, -0.5F }, // 3 : 5

		// Back Face
		{-0.5F,  0.5F,  0.5F }, // 5 : 7
		{ 0.5F,  0.5F,  0.5F }, // 4 : 8
		{ 0.5F, -0.5F,  0.5F }, // 8 : 10
		{-0.5F, -0.5F,  0.5F }, // 7 : 11

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
	static const MeshVector3D TemporalNormals{
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
	static const MeshUVs      TemporalTextureCoords{
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
	static const MeshColors   TemporalColors{
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

	TemporalMesh = Mesh(TemporalTriangles, TemporalVertices, TemporalNormals, TemporalTextureCoords, TemporalColors);

	return TemporalMesh;
}

void Mesh::BindVertexArray() {
	// Generate 1 VAO, put the resulting identifier in VAO identifier
	if (VertexArrayObject == 0) glGenVertexArrays(1, &VertexArrayObject);
	// The following commands will put in context our VAO for the next commands
	glBindVertexArray(VertexArrayObject);
}

void Mesh::DrawInstanciated(int Count) const {
	glDrawElementsInstanced(
		GL_TRIANGLES,	                        // mode
		(int)Triangles.size() * 3,	            // mode count
		GL_UNSIGNED_INT,                        // type
		(void*)0,		                        // element array buffer offset
		Count                                   // element count
	);
}

void Mesh::DrawElement() const {
	glDrawElements(
		GL_TRIANGLES,	                        // mode
		(int)Triangles.size() * 3,	            // mode count
		GL_UNSIGNED_INT,                        // type
		(void*)0		                        // element array buffer offset
	); // Starting from vertex 0; to vertices total
}

void Mesh::SetUpBuffers() {

	BindVertexArray();

	// This will identify our vertex buffer
	glGenBuffers(1, &VertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, Vertices.size() * sizeof(Vertex), &Vertices[0], GL_STATIC_DRAW);

	// Generate a Element Buffer for the indices
	glGenBuffers(1, &ElementBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ElementBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, Triangles.size() * sizeof(IntVector3), &Triangles[0], GL_STATIC_DRAW);

	// Set the vertex attribute pointers layouts
	glEnableVertexAttribArray(  VertexLocation );
	    glVertexAttribPointer(  VertexLocation , 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	glEnableVertexAttribArray(  NormalLocation );
	    glVertexAttribPointer(  NormalLocation , 3, GL_FLOAT,  GL_TRUE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
	glEnableVertexAttribArray( TangentLocation );
	    glVertexAttribPointer( TangentLocation , 3, GL_FLOAT,  GL_TRUE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
	glEnableVertexAttribArray(     UV0Location );
	    glVertexAttribPointer(     UV0Location , 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, UV0));
	glEnableVertexAttribArray(     UV1Location );
	    glVertexAttribPointer(     UV1Location , 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, UV1));
	glEnableVertexAttribArray(   ColorLocation );
	    glVertexAttribPointer(   ColorLocation , 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Color));

	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}

void Mesh::ClearBuffers() {
	glDeleteVertexArrays(1, &VertexArrayObject);
}
