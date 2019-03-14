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

#include "..\include\Contour.h"

// Vector shape representation.
class Shape {
public:
	//* The list of contours the shape consists of.
	TArray<Contour> contours;

	//* Specifies whether the shape uses bottom-to-top (false) or top-to-bottom (true) Y coordinates.
	bool inverseYAxis;

	//* Default constructor
	Shape();
	
	//* Adds a contour.
	void addContour(const Contour &contour);

	//* Adds a blank contour and returns its reference.
	Contour & addContour();
	
	//* Normalizes the shape geometry for distance field generation.
	void normalize();
	
	//* Performs basic checks to determine if the object represents a valid shape.
	bool validate() const;
	
	//* Computes the shape's bounding box.
	void bounds(float &l, float &b, float &r, float &t) const;

};