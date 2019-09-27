
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/Mesh.h"
#include "Fonts/SDFGenerator.h"
#include "Fonts/Glyph.h"

namespace EmptySource {

	FontGlyph::FontGlyph() {
		UnicodeValue = 0;
		Bearing = 0;
		Width = 0;
		Height = 0;
		Advance = 0;
		bUndefined = true;
		UV = { 0, 0, 0, 0 };
		VectorShape = Shape2D();
		SDFResterized = PixelMap();
	}

	FontGlyph::FontGlyph(const FontGlyph & Other) :
		UnicodeValue(Other.UnicodeValue), Width(Other.Width), Height(Other.Height),
		Bearing(Other.Bearing), Advance(Other.Advance), UV(Other.UV) {
		VectorShape = Other.VectorShape;
		SDFResterized = Other.SDFResterized;
		bUndefined = Other.bUndefined;
	}

	void FontGlyph::GenerateSDF(float PixelRange) {
		IntVector2 Size = { (int)Width, (int)Height };
		SDFResterized = PixelMap(Size.x + (int)PixelRange * 4, Size.y + (int)PixelRange * 4, 1, PF_R32F);
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

	void FontGlyph::GetQuadMesh(Vector2 Pivot, const float& PixelRange, const float& Scale, const Vector4& Color, MeshVertex * Quad) {
		float XPos = Pivot.x + Bearing.x * Scale;
		float YPos = Pivot.y - ((int)SDFResterized.GetHeight() - Bearing.y) * Scale;
		float XPosWidth = XPos + ((int)SDFResterized.GetWidth() + PixelRange) * Scale;
		float YPosHeight = YPos + ((int)SDFResterized.GetHeight() + PixelRange) * Scale;

		Quad[0].Position.x = XPosWidth; Quad[0].Position.y = YPos;
		Quad[0].UV0.u = UV.xMax; Quad[0].UV0.v = UV.yMin;
		Quad[0].Color = 1.F;

		Quad[1].Position.x = XPos;      Quad[1].Position.y = YPosHeight;
		Quad[1].UV0.u = UV.xMin; Quad[1].UV0.v = UV.yMax;
		Quad[1].Color = 1.F;

		Quad[2].Position.x = XPosWidth; Quad[2].Position.y = YPosHeight;
		Quad[2].UV0.u = UV.xMax; Quad[2].UV0.v = UV.yMax;
		Quad[2].Color = 1.F;

		Quad[3].Position.x = XPos;      Quad[3].Position.y = YPos;
		Quad[3].UV0.u = UV.xMin; Quad[3].UV0.v = UV.yMin;
		Quad[3].Color = 1.F;
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

}