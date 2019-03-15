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

#include "..\include\Shape2D.h"

Shape::Shape() : inverseYAxis(false) { }

void Shape::addContour(const Contour &contour) {
	contours.push_back(contour);
}

Contour & Shape::addContour() {
	contours.resize(contours.size() + 1);
	return contours[contours.size() - 1];
}

bool Shape::validate() const {
	for (std::vector<Contour>::const_iterator contour = contours.begin(); contour != contours.end(); ++contour) {
		if (!contour->edges.empty()) {
			Point2 corner = (*(contour->edges.end() - 1))->point(1);
			for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
				if (!*edge)
					return false;
				if ((*edge)->point(0) != corner)
					return false;
				corner = (*edge)->point(1);
			}
		}
	}
	return true;
}

void Shape::normalize() {
	for (std::vector<Contour>::iterator contour = contours.begin(); contour != contours.end(); ++contour)
		if (contour->edges.size() == 1) {
			EdgeSegment *parts[3] = { };
			contour->edges[0]->splitInThirds(parts[0], parts[1], parts[2]);
			contour->edges.clear();
			contour->edges.push_back(EdgeHolder(parts[0]));
			contour->edges.push_back(EdgeHolder(parts[1]));
			contour->edges.push_back(EdgeHolder(parts[2]));
		}
}

void Shape::bounds(Box2D & Bounds) const {
	for (std::vector<Contour>::const_iterator contour = contours.begin(); contour != contours.end(); ++contour)
		contour->bounds(Bounds.Left, Bounds.Bottom, Bounds.Right, Bounds.Top);
}
