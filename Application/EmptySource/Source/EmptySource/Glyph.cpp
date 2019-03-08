#include "include\Glyph.h"

TextGlyph::TextGlyph() {
	UnicodeValue = 0;
	Size = 0;
	Bearing = 0;
	Data = NULL;
}

TextGlyph::TextGlyph(unsigned long Character, IntVector2 GlyphHeight, IntVector2 GlyphOffset, int GlyphAdvance) {
	UnicodeValue = Character;
	Size = GlyphHeight;
	Bearing = GlyphOffset;
	Advance = GlyphAdvance;
	Data = NULL;
}

TextGlyph::~TextGlyph() {
	delete[] Data;
	Data = NULL;
}
