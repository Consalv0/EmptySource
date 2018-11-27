

#include "..\include\SMath.h"
#include "..\include\SMesh.h"

SMesh::SMesh() {
	Triangles = vector<IVector3>();
	Vertices = Normals = Tangents = vector<FVector3>();
	TextureCoords = vector<FVector2>();
	Colors = vector<FVector4>();
}

SMesh::SMesh(
	const vector<IVector3> InTriangles, const vector<FVector3> InVertices,
	const vector<FVector3> InNormals,   const vector<FVector2> InTextureCoords, const vector<FVector4> InColors) 
{
	    Triangles =     InTriangles; 
	     Vertices =      InVertices; 
	      Normals =       InNormals;
	TextureCoords = InTextureCoords;
	       Colors =        InColors;
		 Tangents = vector<FVector3>();
}
