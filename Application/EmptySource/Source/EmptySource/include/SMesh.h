#pragma once

#include <vector>
using std::vector;

struct IVector3;
struct FVector2;
struct FVector3;
struct FVector4;

class SMesh {
public:
	vector<IVector3> Triangles;
	vector<FVector3> Vertices, Normals, Tangents;
	vector<FVector2> TextureCoords;
	vector<FVector4> Colors;

	SMesh();
	SMesh(
		const vector<IVector3> Triangles, const vector<FVector3> Vertices,
		const vector<FVector3> Normals, const vector<FVector2> TextureCoords, const vector<FVector4> Colors
	);

private:
};