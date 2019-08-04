#pragma once

#include "../include/Math/MathUtility.h"
#include "../include/Math/IntVector2.h"
#include "../include/Shape2D.h"
#include "../include/Bitmap.h"

namespace EmptySource {

	class SDFGenerator {
	private:
		struct Pixel {
			float Alpha, Distance;
			Vector2 Gradient;
			IntVector2 Delta;
		};

		static int Width, Height;
		static Pixel * Pixels;

		static inline Pixel * PixelAt(int x, int y) { return &Pixels[x + Width * y]; };

		static void ComputeEdgeGradients();

		static float ApproximateEdgeDelta(float gx, float gy, float a);

		static void UpdateDistance(Pixel * p, int x, int y, int oX, int oY);

		static void GenerateDistanceTransform();

	public:
		static void FromShape(Bitmap<FloatRed> &Output, const Shape2D &Shape, double Range, const Vector2 &Scale, const Vector2 &Translate);

		static void FromBitmap(Bitmap<FloatRed> &Output, Bitmap<FloatRed> &Input, float MaxInside, float MaxOutside);
	};

}