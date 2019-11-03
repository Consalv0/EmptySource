
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/Mesh.h"
#include "Fonts/SDFGenerator.h"
#include "Fonts/Glyph.h"

namespace ESource {

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
		SDFResterized = PixelMap(Size.X + (int)PixelRange * 4, Size.Y + (int)PixelRange * 4, 1, PF_R32F);
		Vector2 Translate(-(float)Bearing.X, Height - (float)Bearing.Y);
		Vector2 Scale = 1;
		Translate += PixelRange;

		// // --- Auto-frame
		// Vector2 FrameSize = Size.FloatVector2();
		// FrameSize -= 2 * PixelRange;
		// 
		// if (Bounds.Left >= Bounds.Right || Bounds.Bottom >= Bounds.Top)
		// 	Bounds.Left = 0, Bounds.Bottom = 0, Bounds.Right = 1, Bounds.Top = 1;
		// if (FrameSize.X <= 0 || FrameSize.Y <= 0)
		// 	return;
		// 
		// Vector2 Dimension( Bounds.Right - Bounds.Left, Bounds.Top - Bounds.Bottom );
		// 
		// if (Dimension.X * FrameSize.Y < Dimension.Y * FrameSize.X) {
		// 	Translate = { .5F * (FrameSize.X / FrameSize.Y * Dimension.Y - Dimension.X) - Bounds.Left, -Bounds.Bottom };
		// 	Scale = FrameSize.Y / Dimension.Y;
		// }
		// else {
		// 	Translate = { -Bounds.Left, .5F * (FrameSize.Y / FrameSize.X * Dimension.X - Dimension.Y) - Bounds.Bottom };
		// 	Scale = FrameSize.X / Dimension.X;
		// }
		// 

		SDFGenerator::FromShape(SDFResterized, VectorShape, PixelRange / Math::Min(Scale.X, Scale.Y), Scale, Translate);
	}

	void FontGlyph::GetQuadMesh(Vector2 Pivot, const float& PixelRange, const float& Scale, const Vector4& Color, StaticVertex * Quad) {
		float XPos = Pivot.X + Bearing.X * Scale;
		float YPos = Pivot.Y - ((int)SDFResterized.GetHeight() - Bearing.Y) * Scale;
		float XPosWidth = XPos + ((int)SDFResterized.GetWidth() + PixelRange) * Scale;
		float YPosHeight = YPos + ((int)SDFResterized.GetHeight() + PixelRange) * Scale;

		Quad[0].Position.X = XPosWidth; Quad[0].Position.Y = YPos;
		Quad[0].UV0.u = UV.MaxX; Quad[0].UV0.v = UV.MinY;
		Quad[0].Color = 1.F;

		Quad[1].Position.X = XPos;      Quad[1].Position.Y = YPosHeight;
		Quad[1].UV0.u = UV.MinX; Quad[1].UV0.v = UV.MaxY;
		Quad[1].Color = 1.F;

		Quad[2].Position.X = XPosWidth; Quad[2].Position.Y = YPosHeight;
		Quad[2].UV0.u = UV.MaxX; Quad[2].UV0.v = UV.MaxY;
		Quad[2].Color = 1.F;

		Quad[3].Position.X = XPos;      Quad[3].Position.Y = YPos;
		Quad[3].UV0.u = UV.MinX; Quad[3].UV0.v = UV.MinY;
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