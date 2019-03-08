#pragma once

#include "..\include\Math\Math.h"

struct TextGlyph {
public:
	unsigned long UnicodeValue;
	IntVector2 Size;
	IntVector2 Bearing;
	int Advance;
	Vector2 MinUV;
	Vector2 MaxUV;
	unsigned char * Data;

	TextGlyph();

	TextGlyph(unsigned long Character, IntVector2 Size, IntVector2 Bearing, int Advance);

	~TextGlyph();
};