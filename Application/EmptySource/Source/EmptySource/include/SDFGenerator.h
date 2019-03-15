#pragma once

#include "..\include\Math\Math.h"
#include "..\include\Shape.h"
#include "..\include\Bitmap.h"

class SDFGenerator {
public:
	static void Generate(Bitmap<float> &Output, const Shape &shape, double Range, const Vector2 &Scale, const Vector2 &Translate);
};

class SDFTextureGenerator {
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
	static void Generate(Bitmap<unsigned char> &Output, float MaxInside, float MaxOutside);
};