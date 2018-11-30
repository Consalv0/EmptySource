#pragma once

#include <vector>
using std::vector;

struct IVector3;
struct FVector2;
struct FVector3;
struct FVector4;

struct Vertex {
	FVector3 Position;
	FVector3 Normal;
	FVector3 Tangent;
	FVector2 UV0, UV1;
	FVector4 Color;
};

typedef vector<IVector3> SMeshTriangles;
typedef vector<FVector3> SMeshVector3D;
typedef vector<FVector2> SMeshUVs;
typedef vector<FVector4> SMeshColors;
typedef vector<Vertex> SMeshVertices;

class SMesh {
public:
	SMeshTriangles Triangles;
	SMeshVertices Vertices;

	SMesh();
	SMesh(const SMeshTriangles Triangles, const SMeshVector3D Vertices, const SMeshVector3D Normals, const SMeshUVs UV0, const SMeshColors Colors);
	SMesh(const SMeshTriangles Triangles, const SMeshVector3D Vertices, const SMeshVector3D Normals, const SMeshVector3D Tangents, const SMeshUVs UV0, const SMeshColors Colors);

private:
};