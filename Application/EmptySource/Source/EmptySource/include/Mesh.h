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

struct IntVector3;
struct Vector2;
struct Vector3;
struct Vector4;

struct Vertex {
	Vector3 Position;
	Vector3 Normal;
	Vector3 Tangent;
	Vector2 UV0, UV1;
	Vector4 Color;
};

typedef vector<IntVector3> MeshTriangles;
typedef vector<Vector3>    MeshVector3D;
typedef vector<Vector2>    MeshUVs;
typedef vector<Vector4>    MeshColors;
typedef vector<Vertex>     MeshVertices;

class Mesh {
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
	MeshTriangles Triangles;
	MeshVertices Vertices;

	Mesh();
	Mesh(const MeshTriangles Triangles, const MeshVector3D Vertices, const MeshVector3D Normals, const MeshUVs UV0, const MeshColors Colors);
	Mesh(const MeshTriangles Triangles, const MeshVector3D Vertices, const MeshVector3D Normals, const MeshVector3D Tangents, const MeshUVs UV0, const MeshColors Colors);
	Mesh(const MeshTriangles Triangles, const MeshVector3D Vertices, const MeshVector3D Normals, const MeshVector3D Tangents, const MeshUVs UV0, const MeshUVs UV1, const MeshColors Colors);
	
	static Mesh BuildCube();

	//* *Create or Bind Vertex Array Object
	void BindVertexArray();

	//* Draw mesh using instanciated Element Buffer
	void DrawInstanciated(int Count) const;

	//* Draw mesh using Element Buffer
	void DrawElement() const;
};