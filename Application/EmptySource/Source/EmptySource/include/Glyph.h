#pragma once

#include "..\include\Math\Math.h"

struct TextGlyph {
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

	TextGlyph();

	TextGlyph(unsigned long Character, IntVector2 Size, IntVector2 Bearing, int Advance);

	void GetQuadMesh(Vector2 Pivot, struct MeshVertex * Quad);

	~TextGlyph();
};