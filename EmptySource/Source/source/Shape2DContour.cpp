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

#include "../include/Math/MathUtility.h"
#include "../include/Shape2DContour.h"

void Shape2DContour::AddEdge(const EdgeHolder &Edge) {
	Edges.push_back(Edge);
}

EdgeHolder & Shape2DContour::AddEdge() {
	Edges.resize(Edges.size() + 1);
	return Edges[Edges.size() - 1];
}

void Shape2DContour::GetBounds(float &Left, float &Bottom, float &Right, float &Top) const {
	for (TArray<EdgeHolder>::const_iterator Edge = Edges.begin(); Edge != Edges.end(); ++Edge)
		(*Edge)->GetBounds(Left, Bottom, Right, Top);
}

void Shape2DContour::GetBounds(BoundingBox2D & BBox) const {
	for (TArray<EdgeHolder>::const_iterator Edge = Edges.begin(); Edge != Edges.end(); ++Edge)
		(*Edge)->GetBounds(BBox.Left, BBox.Bottom, BBox.Right, BBox.Top);
}

int Shape2DContour::Winding() const {
	if (Edges.empty())
		return 0;
	float Total = 0;
	if (Edges.size() == 1) {
		Point2 a = Edges[0]->PointAt(0), b = Edges[0]->PointAt(1 / 3.F), c = Edges[0]->PointAt(2 / 3.F);
		Total += MathEquations::Shoelace2(a, b);
		Total += MathEquations::Shoelace2(b, c);
		Total += MathEquations::Shoelace2(c, a);
	}
	else if (Edges.size() == 2) {
		Point2 a = Edges[0]->PointAt(0), b = Edges[0]->PointAt(.5F), c = Edges[1]->PointAt(0), d = Edges[1]->PointAt(.5F);
		Total += MathEquations::Shoelace2(a, b);
		Total += MathEquations::Shoelace2(b, c);
		Total += MathEquations::Shoelace2(c, d);
		Total += MathEquations::Shoelace2(d, a);
	}
	else {
		Point2 Previous = Edges[Edges.size() - 1]->PointAt(0);
		for (TArray<EdgeHolder>::const_iterator edge = Edges.begin(); edge != Edges.end(); ++edge) {
			Point2 Current = (*edge)->PointAt(0);
			Total += MathEquations::Shoelace2(Previous, Current);
			Previous = Current;
		}
	}
	return Math::Sign(Total);
}
