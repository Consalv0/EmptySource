#pragma once

#include <vector>
using std::vector;

struct IVector3;
struct FVector2;
struct FVector3;
struct FVector4;

typedef vector<IVector3> SMeshTriangles;
typedef vector<FVector3> SMeshVector3D;
typedef vector<FVector2> SMeshUVs;
typedef vector<FVector4> SMeshColors;

class SMesh {
public:
	SMeshTriangles Triangles;
	SMeshVector3D  Vertices; 
	SMeshVector3D  Normals;
	SMeshVector3D  Tangents;
	SMeshUVs       UV0;
	SMeshColors    Colors;

	SMesh();
	SMesh(const SMeshTriangles Triangles, const SMeshVector3D Vertices, const SMeshVector3D Normals, const SMeshUVs UV0, const SMeshColors Colors);
	SMesh(const SMeshTriangles Triangles, const SMeshVector3D Vertices, const SMeshVector3D Normals, const SMeshVector3D Tangents, const SMeshUVs UV0, const SMeshColors Colors);

private:
};