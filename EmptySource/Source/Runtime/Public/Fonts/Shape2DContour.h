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

#pragma once

#include "Math/Box2D.h"
#include "Fonts/EdgeHolder.h"

namespace ESource {

	// A single closed contour of a shape.
	class Shape2DContour {
	public:
		// The sequence of edges that make up the contour.
		TArray<EdgeHolder> Edges;

		// Adds an edge to the contour.
		void AddEdge(const EdgeHolder & Edge);

		// Creates a new edge in the contour and returns its reference.
		EdgeHolder & AddEdge();

		// Computes the bounding box of the contour.
		void GetBounds(float &Left, float &Bottom, float &Right, float &Top) const;

		// Computes the bounding box of the contour.
		void GetBounds(BoundingBox2D & BBox) const;

		// Computes the winding of the contour. Returns 1 if positive, -1 if negative.
		int Winding() const;
	};

}