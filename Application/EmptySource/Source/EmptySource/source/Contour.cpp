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

#include "..\include\Contour.h"
#include "..\include\Math\MathUtility.h"

static float shoelace(const Point2 &a, const Point2 &b) {
	return (b.x - a.x)*(a.y + b.y);
}

void Contour::addEdge(const EdgeHolder &edge) {
	edges.push_back(edge);
}

EdgeHolder & Contour::addEdge() {
	edges.resize(edges.size() + 1);
	return edges[edges.size() - 1];
}

void Contour::bounds(float &l, float &b, float &r, float &t) const {
	for (TArray<EdgeHolder>::const_iterator edge = edges.begin(); edge != edges.end(); ++edge)
		(*edge)->bounds(l, b, r, t);
}

int Contour::winding() const {
	if (edges.empty())
		return 0;
	float Total = 0;
	if (edges.size() == 1) {
		Point2 a = edges[0]->point(0), b = edges[0]->point(1 / 3.F), c = edges[0]->point(2 / 3.F);
		Total += shoelace(a, b);
		Total += shoelace(b, c);
		Total += shoelace(c, a);
	}
	else if (edges.size() == 2) {
		Point2 a = edges[0]->point(0), b = edges[0]->point(.5F), c = edges[1]->point(0), d = edges[1]->point(.5F);
		Total += shoelace(a, b);
		Total += shoelace(b, c);
		Total += shoelace(c, d);
		Total += shoelace(d, a);
	}
	else {
		Point2 prev = edges[edges.size() - 1]->point(0);
		for (std::vector<EdgeHolder>::const_iterator edge = edges.begin(); edge != edges.end(); ++edge) {
			Point2 cur = (*edge)->point(0);
			Total += shoelace(prev, cur);
			prev = cur;
		}
	}
	return Math::Sign(Total);
}