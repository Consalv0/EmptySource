#pragma once

#include "..\include\Math\Math.h"

struct FontGlyph {
public:
	unsigned long UnicodeValue;
	IntVector2 Size;
	IntVector2 Bearing;
	int Advance;
	float MinU;
	float MaxU;
	float MinV;
	float MaxV;
	unsigned char * RasterizedData;

	FontGlyph();

	FontGlyph(unsigned long Character, IntVector2 Size, IntVector2 Bearing, int Advance, unsigned char * Data);

	FontGlyph(const FontGlyph & Other);

	void GetQuadMesh(Vector2 Pivot, const float& Scale, struct MeshVertex * Quad);

	FontGlyph & operator=(const FontGlyph & Other);

	~FontGlyph();
};