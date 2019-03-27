#pragma once

#include "../include/Math/CoreMath.h"
#include "../include/Shape2D.h"
#include "../include/Bitmap.h"

struct FontGlyph {
public:
	unsigned long UnicodeValue;
	float Width;
	float Height;
	IntVector2 Bearing;
	float Advance;
	Box2D UV;
	Shape2D VectorShape;
	Bitmap<float> SDFResterized;

	FontGlyph();

	FontGlyph(const FontGlyph & Other);

	void GenerateSDF(float PixelRange);

	void GetQuadMesh(Vector2 Pivot, const float& PixelRange, const float& Scale, struct MeshVertex * Quad);

	FontGlyph & operator=(const FontGlyph & Other);
};
