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

#include "Fonts/EdgeSegments.h"

namespace EmptySource {

	/// Container for a single edge of dynamic type.
	class EdgeHolder {

	public:
		EdgeHolder();
		EdgeHolder(EdgeSegment *Segment);
		EdgeHolder(Point2 P0, Point2 P1, EdgeColor edgeColor = WHITE);
		EdgeHolder(Point2 P0, Point2 P1, Point2 P2, EdgeColor edgeColor = WHITE);
		EdgeHolder(Point2 P0, Point2 P1, Point2 P2, Point2 P3, EdgeColor edgeColor = WHITE);
		EdgeHolder(const EdgeHolder &Origin);
		~EdgeHolder();

		EdgeHolder & operator=(const EdgeHolder &Origin);
		EdgeSegment & operator*();
		const EdgeSegment & operator*() const;
		EdgeSegment * operator->();
		const EdgeSegment * operator->() const;
		operator EdgeSegment *();
		operator const EdgeSegment *() const;

	private:
		EdgeSegment *edgeSegment;

	};

}