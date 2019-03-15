#pragma once

#include "..\include\Math\Math.h"
#include "..\include\Shape2D.h"
#include "..\include\Bitmap.h"

struct FontGlyph {
public:
	unsigned long UnicodeValue;
	IntVector2 Bearing;
	float Advance;
	float MinU;
	float MaxU;
	float MinV;
	float MaxV;
	Shape VectorShape;
	Bitmap<float> SDFResterized;

	FontGlyph();

	FontGlyph(unsigned long Character, IntVector2 Bearing, float Advance, Shape VectorShape);

	FontGlyph(const FontGlyph & Other);

	void GenerateSDF(const IntVector2 & Size);

	void GetQuadMesh(Vector2 Pivot, const float& Scale, struct MeshVertex * Quad);

	FontGlyph & operator=(const FontGlyph & Other);
};