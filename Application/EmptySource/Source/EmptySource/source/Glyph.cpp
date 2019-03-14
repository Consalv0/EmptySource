#include "..\include\Glyph.h"
#include "..\include\Mesh.h"

FontGlyph::FontGlyph() {
	UnicodeValue = 0;
	Size = 0;
	Bearing = 0;
	RasterizedData = NULL;
}

FontGlyph::FontGlyph(unsigned long Character, IntVector2 Size, IntVector2 Bearing, int Advance, unsigned char * Data) :
	UnicodeValue(Character), Size(Size), Bearing(Bearing), Advance(Advance) {

	if (Data) {
		RasterizedData = new unsigned char[Size.x * Size.y];
		memcpy(RasterizedData, Data, Size.x * Size.y * sizeof(unsigned char));
	}
	else {
		RasterizedData = NULL;
	}
}

FontGlyph::FontGlyph(const FontGlyph & Other) :
	UnicodeValue(Other.UnicodeValue), Size(Other.Size), Bearing(Other.Bearing), Advance(Other.Advance) {

	if (Other.RasterizedData) {
		RasterizedData = new unsigned char[Size.x * Size.y];
		memcpy(RasterizedData, Other.RasterizedData, Size.x * Size.y * sizeof(unsigned char));
	}
	else {
		RasterizedData = NULL;
	}
}

void FontGlyph::GetQuadMesh(Vector2 Pivot, const float& Scale, MeshVertex * Quad) {
	float XPos = Pivot.x + Bearing.x * Scale;
	float YPos = Pivot.y - (Size.y - Bearing.y) * Scale;
	float XPosWidth = XPos + Size.x * Scale;
	float YPosHeight = YPos + Size.y * Scale;

	Quad[0].Position.x = XPosWidth; Quad[0].Position.y = YPos;
	Quad[0].UV0.u = MaxU; Quad[0].UV0.v = MinV;

	Quad[1].Position.x = XPos;      Quad[1].Position.y = YPosHeight;
	Quad[1].UV0.u = MinU; Quad[1].UV0.v = MaxV;
	
	Quad[2].Position.x = XPosWidth; Quad[2].Position.y = YPosHeight;
	Quad[2].UV0.u = MaxU; Quad[2].UV0.v = MaxV;
	
	Quad[3].Position.x = XPos;      Quad[3].Position.y = YPos;
	Quad[3].UV0.u = MinU; Quad[3].UV0.v = MinV;
}

FontGlyph & FontGlyph::operator=(const FontGlyph & Other) {
	if (RasterizedData) {
		delete[] RasterizedData;
	}

	UnicodeValue = Other.UnicodeValue;
	Size = Other.Size;
	Bearing = Other.Bearing;
	Advance = Other.Advance;

	if (Other.RasterizedData) {
		RasterizedData = new unsigned char[Size.x * Size.y];
		memcpy(RasterizedData, Other.RasterizedData, Size.x * Size.y * sizeof(unsigned char));
	}
	else {
		RasterizedData = NULL;
	}

	return *this;
}

FontGlyph::~FontGlyph() {
	delete[] RasterizedData;
	RasterizedData = NULL;
}
