#include "include/Glyph.h"
#include "include/Mesh.h"

TextGlyph::TextGlyph() {
	UnicodeValue = 0;
	Size = 0;
	Bearing = 0;
	RasterizedData = NULL;
}

TextGlyph::TextGlyph(unsigned long Character, IntVector2 GlyphHeight, IntVector2 GlyphOffset, int GlyphAdvance) {
	UnicodeValue = Character;
	Size = GlyphHeight;
	Bearing = GlyphOffset;
	Advance = GlyphAdvance;
	RasterizedData = NULL;
}

void TextGlyph::GetQuadMesh(Vector2 Pivot, MeshVertex * Quad) {
	float XPos = Pivot.x + Bearing.x;
	float YPos = Pivot.y - (Size.y - Bearing.y);
	float XPosWidth = XPos + Size.x;
	float YPosHeight = YPos + Size.y;

	Quad[0].Position.x = XPosWidth;
	Quad[0].Position.y = YPos;
	Quad[0].UV0.u = MaxU;
	Quad[0].UV0.v = MinV;
	Quad[1].Position.x = XPos;
	Quad[1].Position.y = YPosHeight;
	Quad[1].UV0.u = MinU;
	Quad[1].UV0.v = MaxV;
	Quad[2].Position.x = XPosWidth;
	Quad[2].Position.y = YPosHeight;
	Quad[2].UV0.u = MaxU;
	Quad[2].UV0.v = MaxV;
	Quad[3].Position.x = XPos;
	Quad[3].Position.y = YPos;
	Quad[3].UV0.u = MinU;
	Quad[3].UV0.v = MinV;
}

TextGlyph::~TextGlyph() {
	delete[] RasterizedData;
	RasterizedData = NULL;
}
