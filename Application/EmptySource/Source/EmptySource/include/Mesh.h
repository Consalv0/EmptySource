#pragma once

// Default vertex Locations
constexpr auto PositionLocation = 0;
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

struct MeshVertex {
	Vector3 Position;
	Vector3 Normal;
	Vector3 Tangent;
	Vector2 UV0, UV1;
	Vector4 Color;

	MeshVertex() = default;
	MeshVertex(const MeshVertex& Other) = default;
	MeshVertex(MeshVertex&& Other) = default;
	MeshVertex(const Vector3& P, const Vector3& N, const Vector2& UV);
	MeshVertex(const Vector3& P, const Vector3& N, const Vector3& T, const Vector2& UV0, const Vector2& UV1, const Vector4& C);
	MeshVertex& operator=(const MeshVertex& other) = default;
	bool operator<(const MeshVertex That) const; 
	bool operator==(const MeshVertex &Other) const;
};

typedef vector<IntVector3> MeshFaces;
typedef vector<Vector3>    MeshVector3D;
typedef vector<Vector2>    MeshUVs;
typedef vector<Vector4>    MeshColors;
typedef vector<MeshVertex> MeshVertices;

class Mesh {
private:
	//* Vertex Array Object 
	unsigned int VertexArrayObject;
	unsigned int VertexBuffer;
	unsigned int ElementBuffer;

public:
	MeshFaces Faces;
	MeshVertices Vertices;

	Mesh();
	Mesh(const MeshFaces Faces, const MeshVector3D Vertices, const MeshVector3D Normals, const MeshUVs UV0, const MeshColors Colors);
	Mesh(const MeshFaces Faces, const MeshVector3D Vertices, const MeshVector3D Normals, const MeshUVs UV0, const MeshUVs UV1, const MeshColors Colors);
	Mesh(const MeshFaces Faces, const MeshVector3D Vertices, const MeshVector3D Normals, const MeshVector3D Tangents, const MeshUVs UV0, const MeshColors Colors);
	Mesh(const MeshFaces Faces, const MeshVector3D Vertices, const MeshVector3D Normals, const MeshVector3D Tangents, const MeshUVs UV0, const MeshUVs UV1, const MeshColors Colors);
	//* Copy the information to the mesh, the data will be coppied
	Mesh(const MeshFaces Faces, const MeshVertices Vertices);
	//* Transfer information to the mesh, the data will be swapped
	Mesh(MeshFaces* Faces, MeshVertices* Vertices);

	//* *Create or Bind Vertex Array Object
	void BindVertexArray() const;

	//* Draw mesh using instanciated Element Buffer
	void DrawInstanciated(int Count) const;

	//* Draw mesh using Element Buffer
	void DrawElement() const;

	//* Clear the mesh entirely
	void Clear();

	//* Clear the GL's objects
	void ClearBuffers();

	//* Give Vertices to OpenGL **This must be done once**
	void SetUpBuffers();
};