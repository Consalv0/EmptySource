#pragma once

#include "Math/MathUtility.h"
#include "Math/IntVector2.h"
#include "Fonts/Shape2D.h"
#include "Rendering/PixelMap.h"

namespace ESource {

	class SDFGenerator {
	private:
		struct Pixel {
			float Alpha, Distance;
			Vector2 Gradient;
			IntVector2 Delta;
		};

		static int Width, Height;
		static Pixel * Pixels;

		static inline Pixel * PixelAt(int X, int Y) { return &Pixels[X + Width * Y]; };

		static void ComputeEdgeGradients();

		static float ApproximateEdgeDelta(float gx, float gy, float A);

		static void UpdateDistance(Pixel * p, int X, int Y, int oX, int oY);

		static void GenerateDistanceTransform();

	public:
		static void FromShape(PixelMap &Output, const Shape2D &Shape, double Range, const Vector2 &Scale, const Vector2 &Translate);

		static void FromBitmap(PixelMap &Output, PixelMap &Input, float MaxInside, float MaxOutside);
	};

}