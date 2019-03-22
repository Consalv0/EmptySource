// Copyright(c) 2016 Viktor Chlumsky
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../include/Shape.h"

Shape::Shape() : bInverseYAxis(false) { }

void Shape::AddContour(const ShapeContour &Contour) {
	Contours.push_back(Contour);
}

ShapeContour & Shape::AddContour() {
	Contours.resize(Contours.size() + 1);
	return Contours[Contours.size() - 1];
}

bool Shape::Validate() const {
	for (TArray<ShapeContour>::const_iterator Contour = Contours.begin(); Contour != Contours.end(); ++Contour) {
		if (!Contour->Edges.empty()) {
			Point2 Corner = (*(Contour->Edges.end() - 1))->PointAt(1);
			for (TArray<EdgeHolder>::const_iterator Edge = Contour->Edges.begin(); Edge != Contour->Edges.end(); ++Edge) {
				if (!*Edge)
					return false;
				if ((*Edge)->PointAt(0) != Corner)
					return false;
				Corner = (*Edge)->PointAt(1);
			}
		}
	}
	return true;
}

void Shape::Normalize() {
	for (TArray<ShapeContour>::iterator Contour = Contours.begin(); Contour != Contours.end(); ++Contour)
		if (Contour->Edges.size() == 1) {
			EdgeSegment *parts[3] = { };
			Contour->Edges[0]->SplitInThirds(parts[0], parts[1], parts[2]);
			Contour->Edges.clear();
			Contour->Edges.push_back(EdgeHolder(parts[0]));
			Contour->Edges.push_back(EdgeHolder(parts[1]));
			Contour->Edges.push_back(EdgeHolder(parts[2]));
		}
}

void Shape::Bounds(Box2D & Bounds) const {
	for (TArray<ShapeContour>::const_iterator Contour = Contours.begin(); Contour != Contours.end(); ++Contour)
		Contour->GetBounds(Bounds.Left, Bounds.Bottom, Bounds.Right, Bounds.Top);
}
