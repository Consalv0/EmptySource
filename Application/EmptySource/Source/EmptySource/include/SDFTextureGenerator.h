#pragma once

#include "..\include\Math\Math.h"
#include "..\include\Bitmap.h"

class SDFGenerator {
public:
	// static void Gnerate(Pixmap<float> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate) {
	// 	int contourCount = shape.contours.size();
	// 	int w = output.GetWidth(), h = output.GetHeight();
	// 	std::vector<int> windings;
	// 	windings.reserve(contourCount);
	// 	for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
	// 		windings.push_back(contour->winding());
	// 	{
	// 		std::vector<double> contourSD;
	// 		contourSD.resize(contourCount);
	// 		for (int y = 0; y < h; ++y) {
	// 			int row = shape.inverseYAxis ? h - y - 1 : y;
	// 			for (int x = 0; x < w; ++x) {
	// 				double dummy;
	// 				Point2 p = Vector2(x + .5, y + .5) / scale - translate;
	// 				double negDist = -SignedDistance::INFINITE.distance;
	// 				double posDist = SignedDistance::INFINITE.distance;
	// 				int winding = 0;
	// 
	// 				std::vector<Contour>::const_iterator contour = shape.contours.begin();
	// 				for (int i = 0; i < contourCount; ++i, ++contour) {
	// 					SignedDistance minDistance;
	// 					for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
	// 						SignedDistance distance = (*edge)->signedDistance(p, dummy);
	// 						if (distance < minDistance)
	// 							minDistance = distance;
	// 					}
	// 					contourSD[i] = minDistance.distance;
	// 					if (windings[i] > 0 && minDistance.distance >= 0 && fabs(minDistance.distance) < fabs(posDist))
	// 						posDist = minDistance.distance;
	// 					if (windings[i] < 0 && minDistance.distance <= 0 && fabs(minDistance.distance) < fabs(negDist))
	// 						negDist = minDistance.distance;
	// 				}
	// 
	// 				double sd = SignedDistance::INFINITE.distance;
	// 				if (posDist >= 0 && fabs(posDist) <= fabs(negDist)) {
	// 					sd = posDist;
	// 					winding = 1;
	// 					for (int i = 0; i < contourCount; ++i)
	// 						if (windings[i] > 0 && contourSD[i] > sd && fabs(contourSD[i]) < fabs(negDist))
	// 							sd = contourSD[i];
	// 				}
	// 				else if (negDist <= 0 && fabs(negDist) <= fabs(posDist)) {
	// 					sd = negDist;
	// 					winding = -1;
	// 					for (int i = 0; i < contourCount; ++i)
	// 						if (windings[i] < 0 && contourSD[i] < sd && fabs(contourSD[i]) < fabs(posDist))
	// 							sd = contourSD[i];
	// 				}
	// 				for (int i = 0; i < contourCount; ++i)
	// 					if (windings[i] != winding && fabs(contourSD[i]) < fabs(sd))
	// 						sd = contourSD[i];
	// 
	// 				output(x, row) = float(sd / range + .5);
	// 			}
	// 		}
	// 	}
	// }
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
	static inline unsigned char & DataPixelAt(unsigned char * Data, int x, int y) { return Data[x + Width * y]; };

	static void ComputeEdgeGradients() {
		for (int y = 1; y < Height - 1; y++) {
			for (int x = 1; x < Width - 1; x++) {
				Pixel * p = PixelAt(x, y);
				if (p->Alpha > 0.F && p->Alpha < 1.F) {
					// estimate gradient of edge pixel using surrounding pixels
					float g =
						- PixelAt(x - 1, y - 1)->Alpha
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

	static float ApproximateEdgeDelta(float gx, float gy, float a) {
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

	static void UpdateDistance(Pixel * p, int x, int y, int oX, int oY) {
		Pixel * neighbor = PixelAt(x + oX, y + oY);
		Pixel * closest = PixelAt(x + oX - neighbor->Delta.x, y + oY - neighbor->Delta.y);

		if (closest->Alpha == 0.F /*|| *closest == *p*/) {
			// neighbor has no closest yet
			// or neighbor's closest is p itself
			return;
		}

		int dX = neighbor->Delta.x - oX;
		int dY = neighbor->Delta.y - oY;
		float distance = sqrtf((float)dX * dX + dY * dY) + ApproximateEdgeDelta((float)dX, (float)dY, closest->Alpha);
		if (distance < p->Distance) {
			p->Distance = distance;
			p->Delta.x = dX;
			p->Delta.y = dY;
		}
	}

	static void GenerateDistanceTransform() {
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

	static void PostProcess(float maxDistance) {
		// adjust distances near edges based on the local edge gradient
		for (int y = 0; y < Height; y++) {
			for (int x = 0; x < Width; x++) {
				Pixel * p = PixelAt(x, y);
				if ((p->Delta.x == 0 && p->Delta.y == 0) || p->Distance >= maxDistance) {
					// ignore edge, inside, and beyond max distance
					continue;
				}

				int
					dX = p->Delta.x,
					dY = p->Delta.y;
				Pixel * closest = PixelAt(x - dX, y - dY);
				Vector2 * g = &closest->Gradient;

				if (g->x == 0.F && g->y == 0.F) {
					// ignore unknown gradients (inside)
					continue;
				}

				// compute hit point offset on gradient inside pixel
				float df = ApproximateEdgeDelta(g->x, g->y, closest->Alpha);
				float t =  dY * g->x - dX * g->y;
				float u = -df * g->x + t * g->y;
				float v = -df * g->y - t * g->x;

				// use hit point to compute distance
				if (abs(u) <= 0.5F && abs(v) <= 0.5F) {
					p->Distance = sqrtf((dX + u) * (dX + u) + (dY + v) * (dY + v));
				}
			}
		}
	}
public:
	static void Generate(
		unsigned char * Source,
		const IntVector2 & SourceSize,
		float MaxInside,
		float MaxOutside,
		float PostProcessDistance) {

		Width = SourceSize.x;
		Height = SourceSize.y;
		Pixels = new Pixel[Width * Height];

		int x, y;
		float Scale;
		if (MaxInside > 0.F) {
			for (y = 0; y < Height; y++) {
				for (x = 0; x < Width; x++) {
					PixelAt(x, y)->Alpha = 1.F - (DataPixelAt(Source, x, y) / 255.F);
				}
			}
			ComputeEdgeGradients();
			GenerateDistanceTransform();
			if (PostProcessDistance > 0.F) {
				PostProcess(PostProcessDistance);
			}
			Scale = 1.F / MaxInside;
			for (y = 0; y < Height; y++) {
				for (x = 0; x < Width; x++) {
					float Alpha = Math::Clamp01(PixelAt(x, y)->Distance * Scale);
					DataPixelAt(Source, x, y) = unsigned char(Alpha * 255);
				}
			}
		}
		if (MaxOutside > 0.F) {
			for (y = 0; y < Height; y++) {
				for (x = 0; x < Width; x++) {
					PixelAt(x, y)->Alpha = DataPixelAt(Source, x, y) / 255.F;
				}
			}
			ComputeEdgeGradients();
			GenerateDistanceTransform();
			if (PostProcessDistance > 0.F) {
				PostProcess(PostProcessDistance);
			}
			Scale = 1.F / MaxOutside;
			if (MaxInside > 0.F) {
				for (y = 0; y < Height; y++) {
					for (x = 0; x < Width; x++) {
						float Alpha = 0.5f + ((DataPixelAt(Source, x, y) / 255.F) -
							Math::Clamp01(PixelAt(x, y)->Distance * Scale)) * 0.5f;
						DataPixelAt(Source, x, y) = unsigned char(Alpha * 255);
					}
				}
			}
			else {
				for (y = 0; y < Height; y++) {
					for (x = 0; x < Width; x++) {
						float Alpha = Math::Clamp01(1.F - PixelAt(x, y)->Distance * Scale);
						DataPixelAt(Source, x, y) = unsigned char(Alpha * 255);
					}
				}
			}
		}

		delete[] Pixels;
	}
};

int SDFTextureGenerator::Width = 0, SDFTextureGenerator::Height = 0;
SDFTextureGenerator::Pixel * SDFTextureGenerator::Pixels = NULL;

struct IntGrid {
	IntVector2 Size;
	IntVector2 * Matrix;

	size_t Index(int x, int y) const { return x + Size.x * y; };

	IntGrid() {
		Size = 0;
		Matrix = NULL;
	}

	IntGrid(const IntVector2& Size) : Size(Size) {
		Matrix = new IntVector2[Size.y * Size.x];
	}

	IntGrid(IntGrid const &) = delete;
	IntGrid & operator=(IntGrid const &) = delete;

	~IntGrid() {
		delete[] Matrix;
		Matrix = NULL;
	}
};

const IntVector2 Inside = { 0, 0 };
const IntVector2 Empty = { 9999, 9999 };

IntVector2 Get(IntGrid &Grid, int x, int y) {
	// --- OPTIMIZATION: you can skip the edge check code if you make your grid 
	// --- have a 1-pixel gutter.
	if (x >= 0 && y >= 0 && x < Grid.Size.x && y < Grid.Size.y)
		return Grid.Matrix[Grid.Index(x, y)];
	else
		return Empty;
}

void Put(IntGrid &Grid, int x, int y, const IntVector2 &Value) {
	Grid.Matrix[Grid.Index(x, y)] = Value;
}

void Compare(IntGrid &Grid, IntVector2 &Value, int x, int y, int offsetx, int offsety) {
	IntVector2 Other = Get(Grid, x + offsetx, y + offsety);
	Other.x += offsetx;
	Other.y += offsety;

	if (Other.MagnitudeSquared() < Value.MagnitudeSquared())
		Value = Other;
}

void GenerateSDF(IntGrid &Grid, const IntVector2& Size) {
	for (int i = 0; i < 1; i++) {
		// --- Pass 0
		for (int y = 0; y < Size.y; y++) {
			for (int x = 0; x < Size.x; x++) {
				IntVector2 p = Get(Grid, x, y);
				Compare(Grid, p, x, y, -1, 0);
				Compare(Grid, p, x, y, 0, -1);
				Compare(Grid, p, x, y, -1, -1);
				Compare(Grid, p, x, y, 1, -1);
				Put(Grid, x, y, p);
			}

			for (int x = Size.x - 1; x >= 0; x--) {
				IntVector2 p = Get(Grid, x, y);
				Compare(Grid, p, x, y, 1, 0);
				Put(Grid, x, y, p);
			}
		}

		// --- Pass 1
		for (int y = Size.y - 1; y >= 0; y--) {
			for (int x = Size.x - 1; x >= 0; x--) {
				IntVector2 p = Get(Grid, x, y);
				Compare(Grid, p, x, y, 1, 0);
				Compare(Grid, p, x, y, 0, 1);
				Compare(Grid, p, x, y, -1, 1);
				Compare(Grid, p, x, y, 1, 1);
				Put(Grid, x, y, p);
			}

			for (int x = 0; x < Size.x; x++) {
				IntVector2 p = Get(Grid, x, y);
				Compare(Grid, p, x, y, -1, 0);
				Put(Grid, x, y, p);
			}
		}
	}
}

void GenerateSDFFromUChar(unsigned char * ImageData, const IntVector2 & Size) {
	unsigned char * ImageDataIerator = ImageData;
	IntGrid GridA = IntGrid(Size);
	IntGrid GridB = IntGrid(Size);

	for (int y = 0; y < Size.y; y++) {
		for (int x = 0; x < Size.x; x++) {
			// --- Points inside get marked with a dx/dy of zero.
			// --- Points outside get marked with an infinitely large distance.
			if (*(ImageDataIerator++) < 128) {
				Put(GridA, x, y, Inside);
				Put(GridB, x, y, Empty);
			}
			else {
				Put(GridB, x, y, Inside);
				Put(GridA, x, y, Empty);
			}
		}
	}

	GenerateSDF(GridA, Size);
	GenerateSDF(GridB, Size);
	ImageDataIerator = ImageData;

	for (int y = 0; y < Size.y; y++) {
		for (int x = 0; x < Size.x; x++) {
			// --- Calculate the actual distance from the dx/dy
			int DistA = Get(GridA, x, y).MagnitudeSquared();
			int DistB = Get(GridB, x, y).MagnitudeSquared();
			int Distance = DistA - DistB;
	
			// --- Clamp and scale it, just for display purposes.
			int Result = Distance * 8 + 128;
			if (Result < 0) Result = 0;
			if (Result > 255) Result = 255;
			
			*(ImageDataIerator++) = (unsigned char)Result;
		}
	}

	GridA.Index(0, 0);
}