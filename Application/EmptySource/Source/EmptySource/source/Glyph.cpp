
#include "..\include\SDFGenerator.h"
#include "..\include\Mesh.h"
#include "..\include\Glyph.h"

FontGlyph::FontGlyph() {
	UnicodeValue = 0;
	Bearing = 0;
	VectorShape = Shape();
	SDFResterized = Bitmap<float>();
}

FontGlyph::FontGlyph(unsigned long Character, IntVector2 Bearing, float Advance, Shape VectorShape) :
	UnicodeValue(Character), Bearing(Bearing), Advance(Advance) {
	this->VectorShape = VectorShape;
	SDFResterized = Bitmap<float>();
}

FontGlyph::FontGlyph(const FontGlyph & Other) :
	UnicodeValue(Other.UnicodeValue), Bearing(Other.Bearing), Advance(Other.Advance) {
	VectorShape = Other.VectorShape;
	SDFResterized = Other.SDFResterized;
}

void FontGlyph::GenerateSDF(const IntVector2 & Size) {
	SDFResterized = Bitmap<float>(Size.x, Size.y);
	SDFGenerator::Generate(SDFResterized, VectorShape, 4, 1, {0, 5});
}

void FontGlyph::GetQuadMesh(Vector2 Pivot, const float& Scale, MeshVertex * Quad) {
	float XPos = Pivot.x + Bearing.x * Scale;
	float YPos = Pivot.y - (SDFResterized.GetHeight() - Bearing.y) * Scale;
	float XPosWidth = XPos + SDFResterized.GetWidth() * Scale;
	float YPosHeight = YPos + SDFResterized.GetHeight() * Scale;

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
	UnicodeValue = Other.UnicodeValue;
	Bearing = Other.Bearing;
	Advance = Other.Advance;
	VectorShape = Other.VectorShape;
	SDFResterized = Other.SDFResterized;

	return *this;
}
