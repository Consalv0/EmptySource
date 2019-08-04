#pragma once

#include "../include/Math/MathUtility.h"
#include "../include/Math/IntVector2.h"
#include "../include/Math/Vector2.h"
#include "../include/Math/Box2D.h"
#include "../include/Shape2D.h"
#include "../include/Bitmap.h"

namespace EmptySource {

	struct FontGlyph {
	public:
		unsigned long UnicodeValue;
		float Width;
		float Height;
		IntVector2 Bearing;
		float Advance;
		Box2D UV;
		Shape2D VectorShape;
		Bitmap<FloatRed> SDFResterized;

		FontGlyph();

		FontGlyph(const FontGlyph & Other);

		void GenerateSDF(float PixelRange);

		void GetQuadMesh(Vector2 Pivot, const float& PixelRange, const float& Scale, struct MeshVertex * Quad);

		FontGlyph & operator=(const FontGlyph & Other);
	};

}