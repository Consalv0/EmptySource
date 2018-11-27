

#include "..\include\SMath.h"
#include "..\include\SMesh.h"

SMesh::SMesh() {
	Triangles = SMeshTriangles();
	Vertices = Normals = Tangents = SMeshVector3D();
	UV0 = SMeshUVs();
	Colors = SMeshColors();
}

SMesh::SMesh( 
	const SMeshTriangles triangles, const SMeshVector3D vertices,
	const SMeshVector3D normals, const SMeshUVs textureCoords, const SMeshColors colors)
{
	    Triangles =     triangles; 
	     Vertices =      vertices; 
	      Normals =       normals;
			  UV0 = textureCoords;
	       Colors =        colors;
		 Tangents =   SMeshVector3D();
}

SMesh::SMesh(
	const SMeshTriangles triangles, const SMeshVector3D vertices,
	const SMeshVector3D normals, const SMeshVector3D tangents, const SMeshUVs textureCoords, const SMeshColors colors)
{
	Triangles = triangles;
	Vertices = vertices;
	Normals = normals;
	Tangents = tangents;
	UV0 = textureCoords;
	Colors = colors;
}
