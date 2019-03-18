
#include "..\include\SDFGenerator.h"
#include "..\include\Mesh.h"
#include "..\include\Glyph.h"

FontGlyph::FontGlyph() {
	UnicodeValue = 0;
	Bearing = 0;
	Width = 0;
	Height = 0;
	Advance = 0;
	UV = { 0, 0, 0, 0 };
	VectorShape = Shape();
	SDFResterized = Bitmap<float>();
}

FontGlyph::FontGlyph(const FontGlyph & Other) :
	UnicodeValue(Other.UnicodeValue), Bearing(Other.Bearing), Advance(Other.Advance),
	Width(Other.Width), Height(Other.Height), UV(Other.UV) {
	VectorShape = Other.VectorShape;
	SDFResterized = Other.SDFResterized;
}

void FontGlyph::GenerateSDF(float PixelRange) {
	IntVector2 Size = { (int)Width, (int)Height };
	SDFResterized = Bitmap<float>(Size.x + (int)PixelRange * 4, Size.y + (int)PixelRange * 4);
	Vector2 Translate(-(float)Bearing.x, Height - (float)Bearing.y);
	Vector2 Scale = 1;
	Translate += PixelRange;
 
	// // --- Auto-frame
	// Vector2 FrameSize = Size.FloatVector2();
	// FrameSize -= 2 * PixelRange;
	// 
	// if (Bounds.Left >= Bounds.Right || Bounds.Bottom >= Bounds.Top)
	// 	Bounds.Left = 0, Bounds.Bottom = 0, Bounds.Right = 1, Bounds.Top = 1;
	// if (FrameSize.x <= 0 || FrameSize.y <= 0)
	// 	return;
	// 
	// Vector2 Dimension( Bounds.Right - Bounds.Left, Bounds.Top - Bounds.Bottom );
	// 
	// if (Dimension.x * FrameSize.y < Dimension.y * FrameSize.x) {
	// 	Translate = { .5F * (FrameSize.x / FrameSize.y * Dimension.y - Dimension.x) - Bounds.Left, -Bounds.Bottom };
	// 	Scale = FrameSize.y / Dimension.y;
	// }
	// else {
	// 	Translate = { -Bounds.Left, .5F * (FrameSize.y / FrameSize.x * Dimension.x - Dimension.y) - Bounds.Bottom };
	// 	Scale = FrameSize.x / Dimension.x;
	// }
	// 

	SDFGenerator::FromShape(SDFResterized, VectorShape, PixelRange / Math::Min(Scale.x, Scale.y), Scale, Translate);
}

void FontGlyph::GetQuadMesh(Vector2 Pivot, const float& PixelRange, const float& Scale, MeshVertex * Quad) {
	float XPos = Pivot.x + Bearing.x * Scale;
	float YPos = Pivot.y - (Height - Bearing.y) * Scale;
	float XPosWidth = XPos + (Width + PixelRange) * Scale;
	float YPosHeight = YPos + (Height + PixelRange) * Scale;

	Quad[0].Position.x = XPosWidth; Quad[0].Position.y = YPos;
	Quad[0].UV0.u = UV.xMax; Quad[0].UV0.v = UV.yMin;

	Quad[1].Position.x = XPos;      Quad[1].Position.y = YPosHeight;
	Quad[1].UV0.u = UV.xMin; Quad[1].UV0.v = UV.yMax;
	
	Quad[2].Position.x = XPosWidth; Quad[2].Position.y = YPosHeight;
	Quad[2].UV0.u = UV.xMax; Quad[2].UV0.v = UV.yMax;
	
	Quad[3].Position.x = XPos;      Quad[3].Position.y = YPos;
	Quad[3].UV0.u = UV.xMin; Quad[3].UV0.v = UV.yMin;
}

FontGlyph & FontGlyph::operator=(const FontGlyph & Other) {
	UnicodeValue = Other.UnicodeValue;
	Width = Other.Width;
	Height = Other.Height;
	UV = Other.UV;
	Bearing = Other.Bearing;
	Advance = Other.Advance;
	VectorShape = Other.VectorShape;
	SDFResterized = Other.SDFResterized;

	return *this;
}
