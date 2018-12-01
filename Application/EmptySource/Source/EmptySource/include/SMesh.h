#pragma once

// Vertex Locations
constexpr auto VertexLocation = 0;
constexpr auto NormalLocation = 1;
constexpr auto TangentLocation = 2;
constexpr auto UV0Location = 3;
constexpr auto UV1Location = 4;
constexpr auto ColorLocation = 5;
constexpr auto WeightsLocation = 9;

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
typedef vector<Vertex>   SMeshVertices;

class SMesh {
private:
	//* Vertex Array Object 
	unsigned int VertexArrayObject;
	unsigned int VertexBuffer;
	unsigned int ElementBuffer;

	//* Give Vertices to OpenGL **This must be done once**
	void SetUpBuffers();

	//* Give Vertices to OpenGL **This must be done once**
	void ClearBuffers();

public:
	SMeshTriangles Triangles;
	SMeshVertices Vertices;

	SMesh();
	SMesh(const SMeshTriangles Triangles, const SMeshVector3D Vertices, const SMeshVector3D Normals, const SMeshUVs UV0, const SMeshColors Colors);
	SMesh(const SMeshTriangles Triangles, const SMeshVector3D Vertices, const SMeshVector3D Normals, const SMeshVector3D Tangents, const SMeshUVs UV0, const SMeshColors Colors);
	
	static SMesh BuildCube();

	//* *Create or Bind Vertex Array Object
	void BindVertexArray();

	//* Draw mesh using instanciated Element Buffer
	void DrawInstanciated(int Count) const;

	//* Draw mesh using Element Buffer
	void DrawElement() const;
};