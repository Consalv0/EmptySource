

#include "..\include\SMath.h"
#include "..\include\SMesh.h"

SMesh::SMesh() {
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
}

SMesh::SMesh(
	const SMeshTriangles triangles, const SMeshVector3D vertices,
	const SMeshVector3D normals, const SMeshVector3D tangents, const SMeshUVs uv0, const SMeshColors colors)
{
	Triangles = triangles;
	for (int vCount = 0; vCount < vertices.size(); vCount++) {
		Vertices.push_back(Vertex({ vertices[vCount], normals[vCount], tangents[vCount], uv0[vCount], FVector2(), colors[vCount] }));
	}
}
