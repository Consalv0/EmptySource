#pragma once

#include "..\include\Math\Math.h"

struct TextGlyph {
public:
	unsigned long UnicodeValue;
	int Size;
	IntVector2 Dimension;
	IntVector3 Offset;
	unsigned char * Data;

	TextGlyph();

	TextGlyph(unsigned long Character, int Size, IntVector2 Dimension, IntVector3 Offset);

	~TextGlyph();
};