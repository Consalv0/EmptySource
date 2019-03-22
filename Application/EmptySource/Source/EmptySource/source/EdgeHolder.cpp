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

#include "../include/EdgeHolder.h"

EdgeHolder::EdgeHolder() 
	: edgeSegment(NULL) { }

EdgeHolder::EdgeHolder(EdgeSegment *Segment) 
	: edgeSegment(Segment) { }

EdgeHolder::EdgeHolder(Point2 P0, Point2 P1, EdgeColor edgeColor) 
	: edgeSegment(new LinearSegment(P0, P1, edgeColor)) { }

EdgeHolder::EdgeHolder(Point2 P0, Point2 P1, Point2 P2, EdgeColor edgeColor) 
	: edgeSegment(new QuadraticSegment(P0, P1, P2, edgeColor)) { }

EdgeHolder::EdgeHolder(Point2 P0, Point2 P1, Point2 P2, Point2 P3, EdgeColor edgeColor) 
	: edgeSegment(new CubicSegment(P0, P1, P2, P3, edgeColor)) { }

EdgeHolder::EdgeHolder(const EdgeHolder &Origin) 
	: edgeSegment(Origin.edgeSegment ? Origin.edgeSegment->Clone() : NULL) { }

EdgeHolder::~EdgeHolder() {
    delete edgeSegment;
}

EdgeHolder & EdgeHolder::operator=(const EdgeHolder &Origin) {
    delete edgeSegment;
    edgeSegment = Origin.edgeSegment ? Origin.edgeSegment->Clone() : NULL;
    return *this;
}

EdgeSegment & EdgeHolder::operator*() {
    return *edgeSegment;
}

const EdgeSegment & EdgeHolder::operator*() const {
    return *edgeSegment;
}

EdgeSegment * EdgeHolder::operator->() {
    return edgeSegment;
}

const EdgeSegment * EdgeHolder::operator->() const {
    return edgeSegment;
}

EdgeHolder::operator EdgeSegment *() {
    return edgeSegment;
}

EdgeHolder::operator const EdgeSegment *() const {
    return edgeSegment;
}
