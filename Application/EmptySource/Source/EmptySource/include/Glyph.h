#pragma once

#include "..\include\Math\Math.h"
#include "..\include\Shape.h"
#include "..\include\Bitmap.h"

struct FontGlyph {
public:
	unsigned long UnicodeValue;
	float Width;
	float Height;
	IntVector2 Bearing;
	float Advance;
	Box2D UV;
	Shape VectorShape;
	Bitmap<float> SDFResterized;

	FontGlyph();

	FontGlyph(const FontGlyph & Other);

	void GenerateSDF(float PixelRange = 2.F);

	void GetQuadMesh(Vector2 Pivot, const float& PixelRange, const float& Scale, struct MeshVertex * Quad);

	FontGlyph & operator=(const FontGlyph & Other);
};