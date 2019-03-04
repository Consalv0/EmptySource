#include "include\Glyph.h"

TextGlyph::TextGlyph() {
	UnicodeValue = 0;
	Size = 0;
	Dimension = 0;
	Offset = 0;
	Data = NULL;
}

TextGlyph::TextGlyph(unsigned long Character, int FontSize, IntVector2 GlyphSize, IntVector3 GlyphOffset) {
	UnicodeValue = Character;
	Size = FontSize;
	Dimension = GlyphSize;
	Offset = GlyphOffset;
	Data = NULL;
}

TextGlyph::~TextGlyph() {
	delete[] Data;
	Data = NULL;
}
