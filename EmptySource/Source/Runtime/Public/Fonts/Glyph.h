#pragma once

#include "Math/MathUtility.h"
#include "Math/IntVector2.h"
#include "Math/Vector2.h"
#include "Math/Box2D.h"
#include "Fonts/Shape2D.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/PixelMap.h"

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
		PixelMap SDFResterized;
		bool bUndefined;

		FontGlyph();

		FontGlyph(const FontGlyph & Other);

		void GenerateSDF(float PixelRange);

		void GetQuadMesh(Vector2 Pivot, const float& PixelRange, const float& Scale, const Vector4& Color, struct MeshVertex * Quad);

		FontGlyph & operator=(const FontGlyph & Other);
	};

}