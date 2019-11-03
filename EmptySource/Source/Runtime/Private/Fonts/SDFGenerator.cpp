
#include "CoreMinimal.h"
#include "Rendering/RenderingDefinitions.h"
#include "Fonts/SDFGenerator.h"

namespace ESource {

	int SDFGenerator::Width = 0, SDFGenerator::Height = 0;
	SDFGenerator::Pixel * SDFGenerator::Pixels = NULL;

	void SDFGenerator::ComputeEdgeGradients() {
		for (int Y = 1; Y < Height - 1; Y++) {
			for (int X = 1; X < Width - 1; X++) {
				Pixel * p = PixelAt(X, Y);
				if (p->Alpha > 0.F && p->Alpha < 1.F) {
					// estimate gradient of edge pixel using surrounding pixels
					float G =
						- PixelAt(X - 1, Y - 1)->Alpha
						- PixelAt(X - 1, Y + 1)->Alpha
						+ PixelAt(X + 1, Y - 1)->Alpha
						+ PixelAt(X + 1, Y + 1)->Alpha;
					p->Gradient.X = G + (PixelAt(X + 1, Y)->Alpha - PixelAt(X - 1, Y)->Alpha) * MathConstants::SquareRoot2;
					p->Gradient.Y = G + (PixelAt(X, Y + 1)->Alpha - PixelAt(X, Y - 1)->Alpha) * MathConstants::SquareRoot2;
					p->Gradient.Normalize();
				}
			}
		}
	}

	float SDFGenerator::ApproximateEdgeDelta(float gx, float gy, float A) {
		// (gx, gy) can be either the local pixel gradient or the direction to the pixel

		if (gx == 0.F || gy == 0.F) {
			// linear function is correct if both gx and gy are zero
			// and still fair if only one of them is zero
			return 0.5F - A;
		}

		// normalize (gx, gy)
		float Length = sqrtf(gx * gx + gy * gy);
		gx = gx / Length;
		gy = gy / Length;

		// reduce symmetrical equation to first octant only
		// gx >= 0, gy >= 0, gx >= gy
		gx = abs(gx);
		gy = abs(gy);
		if (gx < gy) {
			float Temp = gx;
			gx = gy;
			gy = Temp;
		}

		// compute delta
		float a1 = 0.5F * gy / gx;
		if (A < a1) {
			// 0 <= a < a1
			return 0.5F * (gx + gy) - sqrtf(2.F * gx * gy * A);
		}
		if (A < (1.F - a1)) {
			// a1 <= a <= 1 - a1
			return (0.5F - A) * gx;
		}
		// 1-a1 < a <= 1
		return -0.5F * (gx + gy) + sqrtf(2.F * gx * gy * (1.F - A));
	}

	void SDFGenerator::UpdateDistance(Pixel * p, int X, int Y, int oX, int oY) {
		Pixel * neighbor = PixelAt(X + oX, Y + oY);
		Pixel * closest = PixelAt(X + oX - neighbor->Delta.X, Y + oY - neighbor->Delta.Y);

		if (closest->Alpha == 0.F /*|| *closest == *p*/) {
			// neighbor has no closest yet
			// or neighbor's closest is p itself
			return;
		}

		int dX = neighbor->Delta.X - oX;
		int dY = neighbor->Delta.Y - oY;
		float Distance = sqrtf((float)dX * dX + dY * dY) + ApproximateEdgeDelta((float)dX, (float)dY, closest->Alpha);
		if (Distance < p->Distance) {
			p->Distance = Distance;
			p->Delta.X = dX;
			p->Delta.Y = dY;
		}
	}

	void SDFGenerator::GenerateDistanceTransform() {
		// perform anti-aliased Euclidean distance transform
		int X, Y;
		Pixel * p;

		// initialize distances
		for (Y = 0; Y < Height; Y++) {
			for (X = 0; X < Width; X++) {
				p = PixelAt(X, Y);
				p->Delta.X = 0;
				p->Delta.Y = 0;
				if (p->Alpha <= 0.F) {
					// outside
					p->Distance = MathConstants::BigNumber;
				}
				else if (p->Alpha < 1.F) {
					// on the edge
					p->Distance = ApproximateEdgeDelta(p->Gradient.X, p->Gradient.Y, p->Alpha);
				}
				else {
					// inside
					p->Distance = 0.F;
				}
			}
		}
		// perform 8SSED (eight-points signed sequential Euclidean distance transform)
		// scan up
		for (Y = 1; Y < Height; Y++) {
			// |P.
			// |XX
			p = PixelAt(0, Y);
			if (p->Distance > 0.F) {
				UpdateDistance(p, 0, Y, 0, -1);
				UpdateDistance(p, 0, Y, 1, -1);
			}
			// -->
			// XP.
			// XXX
			for (X = 1; X < Width - 1; X++) {
				p = PixelAt(X, Y);
				if (p->Distance > 0.F) {
					UpdateDistance(p, X, Y, -1, 0);
					UpdateDistance(p, X, Y, -1, -1);
					UpdateDistance(p, X, Y, 0, -1);
					UpdateDistance(p, X, Y, 1, -1);
				}
			}
			// XP|
			// XX|
			p = PixelAt(Width - 1, Y);
			if (p->Distance > 0.F) {
				UpdateDistance(p, Width - 1, Y, -1, 0);
				UpdateDistance(p, Width - 1, Y, -1, -1);
				UpdateDistance(p, Width - 1, Y, 0, -1);
			}
			// <--
			// .PX
			for (X = Width - 2; X >= 0; X--) {
				p = PixelAt(X, Y);
				if (p->Distance > 0.F) {
					UpdateDistance(p, X, Y, 1, 0);
				}
			}
		}
		// scan down
		for (Y = Height - 2; Y >= 0; Y--) {
			// XX|
			// .P|
			p = PixelAt(Width - 1, Y);
			if (p->Distance > 0.F) {
				UpdateDistance(p, Width - 1, Y, 0, 1);
				UpdateDistance(p, Width - 1, Y, -1, 1);
			}
			// <--
			// XXX
			// .PX
			for (X = Width - 2; X > 0; X--) {
				p = PixelAt(X, Y);
				if (p->Distance > 0.F) {
					UpdateDistance(p, X, Y, 1, 0);
					UpdateDistance(p, X, Y, 1, 1);
					UpdateDistance(p, X, Y, 0, 1);
					UpdateDistance(p, X, Y, -1, 1);
				}
			}
			// |XX
			// |PX
			p = PixelAt(0, Y);
			if (p->Distance > 0.F) {
				UpdateDistance(p, 0, Y, 1, 0);
				UpdateDistance(p, 0, Y, 1, 1);
				UpdateDistance(p, 0, Y, 0, 1);
			}
			// -->
			// XP.
			for (X = 1; X < Width; X++) {
				p = PixelAt(X, Y);
				if (p->Distance > 0.F) {
					UpdateDistance(p, X, Y, -1, 0);
				}
			}
		}
	}

	void SDFGenerator::FromBitmap(PixelMap& Output, PixelMap& Input, float MaxInside, float MaxOutside) {
		Width = Input.GetWidth();
		Height = Input.GetHeight();

		Output = Input;

		if (Pixels != NULL) delete[] Pixels;
		Pixels = new Pixel[Width * Height];

		int X, Y;
		float Scale;
		if (MaxInside > 0.F) {
			for (Y = 0; Y < Height; Y++) {
				for (X = 0; X < Width; X++) {
					PixelAt(X, Y)->Alpha = 1.F - *PixelMapUtility::GetFloatPixelAt(Output, X, Y, 0);
				}
			}
			ComputeEdgeGradients();
			GenerateDistanceTransform();
			Scale = 1.F / MaxInside;
			for (Y = 0; Y < Height; Y++) {
				for (X = 0; X < Width; X++) {
					float Alpha = Math::Clamp01(PixelAt(X, Y)->Distance * Scale);
					*PixelMapUtility::GetFloatPixelAt(Output, X, Y, 0) = Alpha;
				}
			}
		}
		if (MaxOutside > 0.F) {
			for (Y = 0; Y < Height; Y++) {
				for (X = 0; X < Width; X++) {
					PixelAt(X, Y)->Alpha = *PixelMapUtility::GetFloatPixelAt(Output, X, Y, 0);
				}
			}
			ComputeEdgeGradients();
			GenerateDistanceTransform();
			Scale = 1.F / MaxOutside;
			if (MaxInside > 0.F) {
				for (Y = 0; Y < Height; Y++) {
					for (X = 0; X < Width; X++) {
						float Alpha = 0.5f + (*PixelMapUtility::GetFloatPixelAt(Output, X, Y, 0) -
							Math::Clamp01(PixelAt(X, Y)->Distance * Scale)) * 0.5f;
							*PixelMapUtility::GetFloatPixelAt(Output, X, Y, 0) = Alpha;
					}
				}
			}
			else {
				for (Y = 0; Y < Height; Y++) {
					for (X = 0; X < Width; X++) {
						float Alpha = Math::Clamp01(1.F - PixelAt(X, Y)->Distance * Scale);
						*PixelMapUtility::GetFloatPixelAt(Output, X, Y, 0) = Alpha;
					}
				}
			}
		}

		delete[] Pixels;
	}

	void SDFGenerator::FromShape(PixelMap& Output, const Shape2D & Shape, double Range, const Vector2 & Scale, const Vector2 & Translate) {
		int ContourCount = (int)Shape.Contours.size();
		int OutWidth = Output.GetWidth(), OutHeight = Output.GetHeight();
		int * Windings = new int[ContourCount];

		int ShapeContourCount = 0;
		for (TArray<Shape2DContour>::const_iterator Shape2DContour = Shape.Contours.begin(); Shape2DContour != Shape.Contours.end(); ++Shape2DContour)
			Windings[ShapeContourCount++] = Shape2DContour->Winding();
		{
			float * ContourSD = new float[ContourCount];

			for (int Y = 0; Y < OutHeight; ++Y) {
				int Row = Shape.bInverseYAxis ? OutHeight - Y - 1 : Y;
				for (int X = 0; X < OutWidth; ++X) {
					float Dummy;
					Point2 Point = Vector2(X + .5F, Y + .5F) / Scale - Translate;
					float NegDist = MathConstants::BigNumber;
					float PosDist = -MathConstants::BigNumber;
					int Winding = 0;

					TArray<Shape2DContour>::const_iterator Contour = Shape.Contours.begin();
					for (int i = 0; i < ContourCount; ++i, ++Contour) {
						SignedDistance MinDistance;
						for (TArray<EdgeHolder>::const_iterator Edge = Contour->Edges.begin(); Edge != Contour->Edges.end(); ++Edge) {
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

					float SignedDist = -MathConstants::BigNumber;
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

					*PixelMapUtility::GetFloatPixelAt(Output, X, Row, 0) = float(SignedDist / Range + .5F);
				}
			}
			delete[] ContourSD;
		}
		delete[] Windings;
	}

}