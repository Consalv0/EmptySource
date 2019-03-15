#include "..\..\include\SDFGenerator.h"

int SDFTextureGenerator::Width = 0, SDFTextureGenerator::Height = 0;
SDFTextureGenerator::Pixel * SDFTextureGenerator::Pixels = NULL;

void SDFTextureGenerator::ComputeEdgeGradients() {
	for (int y = 1; y < Height - 1; y++) {
		for (int x = 1; x < Width - 1; x++) {
			Pixel * p = PixelAt(x, y);
			if (p->Alpha > 0.F && p->Alpha < 1.F) {
				// estimate gradient of edge pixel using surrounding pixels
				float g =
					-PixelAt(x - 1, y - 1)->Alpha
					- PixelAt(x - 1, y + 1)->Alpha
					+ PixelAt(x + 1, y - 1)->Alpha
					+ PixelAt(x + 1, y + 1)->Alpha;
				p->Gradient.x = g + (PixelAt(x + 1, y)->Alpha - PixelAt(x - 1, y)->Alpha) * MathConstants::SquareRoot2;
				p->Gradient.y = g + (PixelAt(x, y + 1)->Alpha - PixelAt(x, y - 1)->Alpha) * MathConstants::SquareRoot2;
				p->Gradient.Normalize();
			}
		}
	}
}

float SDFTextureGenerator::ApproximateEdgeDelta(float gx, float gy, float a) {
	// (gx, gy) can be either the local pixel gradient or the direction to the pixel

	if (gx == 0.F || gy == 0.F) {
		// linear function is correct if both gx and gy are zero
		// and still fair if only one of them is zero
		return 0.5F - a;
	}

	// normalize (gx, gy)
	float length = sqrtf(gx * gx + gy * gy);
	gx = gx / length;
	gy = gy / length;

	// reduce symmetrical equation to first octant only
	// gx >= 0, gy >= 0, gx >= gy
	gx = abs(gx);
	gy = abs(gy);
	if (gx < gy) {
		float temp = gx;
		gx = gy;
		gy = temp;
	}

	// compute delta
	float a1 = 0.5F * gy / gx;
	if (a < a1) {
		// 0 <= a < a1
		return 0.5F * (gx + gy) - sqrtf(2.F * gx * gy * a);
	}
	if (a < (1.F - a1)) {
		// a1 <= a <= 1 - a1
		return (0.5F - a) * gx;
	}
	// 1-a1 < a <= 1
	return -0.5F * (gx + gy) + sqrtf(2.F * gx * gy * (1.F - a));
}
void SDFTextureGenerator::UpdateDistance(Pixel * p, int x, int y, int oX, int oY) {
	Pixel * neighbor = PixelAt(x + oX, y + oY);
	Pixel * closest = PixelAt(x + oX - neighbor->Delta.x, y + oY - neighbor->Delta.y);

	if (closest->Alpha == 0.F /*|| *closest == *p*/) {
		// neighbor has no closest yet
		// or neighbor's closest is p itself
		return;
	}

	int dX = neighbor->Delta.x - oX;
	int dY = neighbor->Delta.y - oY;
	float Distance = sqrtf((float)dX * dX + dY * dY) + ApproximateEdgeDelta((float)dX, (float)dY, closest->Alpha);
	if (Distance < p->Distance) {
		p->Distance = Distance;
		p->Delta.x = dX;
		p->Delta.y = dY;
	}
}
void SDFTextureGenerator::GenerateDistanceTransform() {
	// perform anti-aliased Euclidean distance transform
	int x, y;
	Pixel * p;

	// initialize distances
	for (y = 0; y < Height; y++) {
		for (x = 0; x < Width; x++) {
			p = PixelAt(x, y);
			p->Delta.x = 0;
			p->Delta.y = 0;
			if (p->Alpha <= 0.F) {
				// outside
				p->Distance = 1000000.F;
			}
			else if (p->Alpha < 1.F) {
				// on the edge
				p->Distance = ApproximateEdgeDelta(p->Gradient.x, p->Gradient.y, p->Alpha);
			}
			else {
				// inside
				p->Distance = 0.F;
			}
		}
	}
	// perform 8SSED (eight-points signed sequential Euclidean distance transform)
	// scan up
	for (y = 1; y < Height; y++) {
		// |P.
		// |XX
		p = PixelAt(0, y);
		if (p->Distance > 0.F) {
			UpdateDistance(p, 0, y, 0, -1);
			UpdateDistance(p, 0, y, 1, -1);
		}
		// -->
		// XP.
		// XXX
		for (x = 1; x < Width - 1; x++) {
			p = PixelAt(x, y);
			if (p->Distance > 0.F) {
				UpdateDistance(p, x, y, -1, 0);
				UpdateDistance(p, x, y, -1, -1);
				UpdateDistance(p, x, y, 0, -1);
				UpdateDistance(p, x, y, 1, -1);
			}
		}
		// XP|
		// XX|
		p = PixelAt(Width - 1, y);
		if (p->Distance > 0.F) {
			UpdateDistance(p, Width - 1, y, -1, 0);
			UpdateDistance(p, Width - 1, y, -1, -1);
			UpdateDistance(p, Width - 1, y, 0, -1);
		}
		// <--
		// .PX
		for (x = Width - 2; x >= 0; x--) {
			p = PixelAt(x, y);
			if (p->Distance > 0.F) {
				UpdateDistance(p, x, y, 1, 0);
			}
		}
	}
	// scan down
	for (y = Height - 2; y >= 0; y--) {
		// XX|
		// .P|
		p = PixelAt(Width - 1, y);
		if (p->Distance > 0.F) {
			UpdateDistance(p, Width - 1, y, 0, 1);
			UpdateDistance(p, Width - 1, y, -1, 1);
		}
		// <--
		// XXX
		// .PX
		for (x = Width - 2; x > 0; x--) {
			p = PixelAt(x, y);
			if (p->Distance > 0.F) {
				UpdateDistance(p, x, y, 1, 0);
				UpdateDistance(p, x, y, 1, 1);
				UpdateDistance(p, x, y, 0, 1);
				UpdateDistance(p, x, y, -1, 1);
			}
		}
		// |XX
		// |PX
		p = PixelAt(0, y);
		if (p->Distance > 0.F) {
			UpdateDistance(p, 0, y, 1, 0);
			UpdateDistance(p, 0, y, 1, 1);
			UpdateDistance(p, 0, y, 0, 1);
		}
		// -->
		// XP.
		for (x = 1; x < Width; x++) {
			p = PixelAt(x, y);
			if (p->Distance > 0.F) {
				UpdateDistance(p, x, y, -1, 0);
			}
		}
	}
}

void SDFTextureGenerator::Generate(Bitmap<unsigned char>& Output, float MaxInside, float MaxOutside) {
	Width = Output.GetWidth();
	Height = Output.GetHeight();

	if (Pixels != NULL) delete[] Pixels;
	Pixels = new Pixel[Width * Height];

	int x, y;
	float Scale;
	if (MaxInside > 0.F) {
		for (y = 0; y < Height; y++) {
			for (x = 0; x < Width; x++) {
				PixelAt(x, y)->Alpha = 1.F - (Output(x, y) / 255.F);
			}
		}
		ComputeEdgeGradients();
		GenerateDistanceTransform();
		Scale = 1.F / MaxInside;
		for (y = 0; y < Height; y++) {
			for (x = 0; x < Width; x++) {
				float Alpha = Math::Clamp01(PixelAt(x, y)->Distance * Scale);
				Output(x, y) = unsigned char(Alpha * 255);
			}
		}
	}
	if (MaxOutside > 0.F) {
		for (y = 0; y < Height; y++) {
			for (x = 0; x < Width; x++) {
				PixelAt(x, y)->Alpha = Output(x, y) / 255.F;
			}
		}
		ComputeEdgeGradients();
		GenerateDistanceTransform();
		Scale = 1.F / MaxOutside;
		if (MaxInside > 0.F) {
			for (y = 0; y < Height; y++) {
				for (x = 0; x < Width; x++) {
					float Alpha = 0.5f + ((Output(x, y) / 255.F) -
						Math::Clamp01(PixelAt(x, y)->Distance * Scale)) * 0.5f;
					Output(x, y) = unsigned char(Alpha * 255);
				}
			}
		}
		else {
			for (y = 0; y < Height; y++) {
				for (x = 0; x < Width; x++) {
					float Alpha = Math::Clamp01(1.F - PixelAt(x, y)->Distance * Scale);
					Output(x, y) = unsigned char(Alpha * 255);
				}
			}
		}
	}

	delete[] Pixels;
}

void SDFGenerator::Generate(Bitmap<float>& Output, const Shape & shape, double Range, const Vector2 & Scale, const Vector2 & Translate) {
	int ContourCount = (int)shape.Contours.size();
	int OutWidth = Output.GetWidth(), OutHeight = Output.GetHeight();
	std::vector<int> Windings;

	Windings.reserve(ContourCount);
	for (std::vector<ShapeContour>::const_iterator ShapeContour = shape.Contours.begin(); ShapeContour != shape.Contours.end(); ++ShapeContour)
		Windings.push_back(ShapeContour->Winding());
	{
		std::vector<float> ContourSD;
		ContourSD.resize(ContourCount);

		for (int y = 0; y < OutHeight; ++y) {
			int Row = shape.bInverseYAxis ? OutHeight - y - 1 : y;
			for (int x = 0; x < OutWidth; ++x) {
				float Dummy;
				Point2 Point = Vector2(x + .5F, y + .5F) / Scale - Translate;
				float NegDist = MathConstants::Big_Number;
				float PosDist = -MathConstants::Big_Number;
				int Winding = 0;

				std::vector<ShapeContour>::const_iterator contour = shape.Contours.begin();
				for (int i = 0; i < ContourCount; ++i, ++contour) {
					SignedDistance MinDistance;
					for (std::vector<EdgeHolder>::const_iterator Edge = contour->Edges.begin(); Edge != contour->Edges.end(); ++Edge) {
						SignedDistance Distance = (*Edge)->GetSignedDistance(Point, Dummy);
						if (Distance < MinDistance)
							MinDistance = Distance;
					}
					ContourSD[i] = MinDistance.Distance;
					if (Windings[i] > 0 && MinDistance.Distance >= 0 && fabs(MinDistance.Distance) < fabs(PosDist))
						PosDist = MinDistance.Distance;
					if (Windings[i] < 0 && MinDistance.Distance <= 0 && fabs(MinDistance.Distance) < fabs(NegDist))
						NegDist = MinDistance.Distance;
				}

				float SignedDist = -MathConstants::Big_Number;
				if (PosDist >= 0 && fabs(PosDist) <= fabs(NegDist)) {
					SignedDist = PosDist;
					Winding = 1;
					for (int i = 0; i < ContourCount; ++i)
						if (Windings[i] > 0 && ContourSD[i] > SignedDist && fabs(ContourSD[i]) < fabs(NegDist))
							SignedDist = ContourSD[i];
				}
				else if (NegDist <= 0 && fabs(NegDist) <= fabs(PosDist)) {
					SignedDist = NegDist;
					Winding = -1;
					for (int i = 0; i < ContourCount; ++i)
						if (Windings[i] < 0 && ContourSD[i] < SignedDist && fabs(ContourSD[i]) < fabs(PosDist))
							SignedDist = ContourSD[i];
				}
				for (int i = 0; i < ContourCount; ++i)
					if (Windings[i] != Winding && fabs(ContourSD[i]) < fabs(SignedDist))
						SignedDist = ContourSD[i];

				Output(x, Row) = float(SignedDist / Range + .5F);
			}
		}
	}
}