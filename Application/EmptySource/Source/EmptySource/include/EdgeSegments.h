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

#include "..\include\Math\Math.h"

// Parameters for iterative search of closest point on a cubic Bezier curve. Increase for higher precision.
#define MSDFGEN_CUBIC_SEARCH_STARTS 4
#define MSDFGEN_CUBIC_SEARCH_STEPS 4

// Edge color specifies which color channels an edge belongs to.
enum EdgeColor {
	BLACK = 0,
	RED = 1,
	GREEN = 2,
	YELLOW = 3,
	BLUE = 4,
	MAGENTA = 5,
	CYAN = 6,
	WHITE = 7
};

// Represents a signed distance and alignment, which together can be compared to uniquely determine the closest edge segment.
class SignedDistance {

public:
	float distance;
	float dot;

	SignedDistance();
	SignedDistance(float Distance, float Dot);

	friend bool operator<(SignedDistance A, SignedDistance B);
	friend bool operator>(SignedDistance A, SignedDistance B);
	friend bool operator<=(SignedDistance A, SignedDistance B);
	friend bool operator>=(SignedDistance A, SignedDistance B);
};

// An abstract edge segment.
class EdgeSegment {

public:
	EdgeColor color;
	
	EdgeSegment(EdgeColor edgeColor = WHITE) : color(edgeColor) { }

	virtual ~EdgeSegment() { }

	// Creates a copy of the edge segment.
	virtual EdgeSegment * clone() const = 0;
	
	// Returns the point on the edge specified by the parameter (between 0 and 1).
	virtual Point2 point(float param) const = 0;
	
	// Returns the direction the edge has at the point specified by the parameter.
	virtual Vector2 direction(float param) const = 0;
	
	// Returns the minimum signed distance between origin and the edge.
	virtual SignedDistance signedDistance(Point2 origin, float &param) const = 0;
	
	// Converts a previously retrieved signed distance from origin to pseudo-distance.
	virtual void distanceToPseudoDistance(SignedDistance &Distance, Point2 Origin, float Param) const;
	
	// Adjusts the bounding box to fit the edge segment.
	virtual void bounds(float &l, float &b, float &r, float &t) const = 0;

	// Moves the start point of the edge segment.
	virtual void moveStartPoint(Point2 To) = 0;
	
	// Moves the end point of the edge segment.
	virtual void moveEndPoint(Point2 To) = 0;
	
	// Splits the edge segments into thirds which together represent the original edge.
	virtual void splitInThirds(EdgeSegment *&part1, EdgeSegment *&part2, EdgeSegment *&part3) const = 0;
};

// A line segment.
class LinearSegment : public EdgeSegment {

public:
	Point2 p[2];

	LinearSegment(Point2 p0, Point2 p1, EdgeColor edgeColor = WHITE);
	LinearSegment * clone() const;
	Point2 point(float Param) const;
	Vector2 direction(float Param) const;
	SignedDistance signedDistance(Point2 Origin, float &Param) const;
	void bounds(float &l, float &b, float &r, float &t) const;

	void moveStartPoint(Point2 To);
	void moveEndPoint(Point2 To);
	void splitInThirds(EdgeSegment *&Part1, EdgeSegment *&Part2, EdgeSegment *&Part3) const;

};

// A quadratic Bezier curve.
class QuadraticSegment : public EdgeSegment {

public:
	Point2 p[3];

	QuadraticSegment(Point2 p0, Point2 p1, Point2 p2, EdgeColor edgeColor = WHITE);
	QuadraticSegment * clone() const;
	Point2 point(float param) const;
	Vector2 direction(float param) const;
	SignedDistance signedDistance(Point2 origin, float &param) const;
	void bounds(float &l, float &b, float &r, float &t) const;

	void moveStartPoint(Point2 To);
	void moveEndPoint(Point2 To);
	void splitInThirds(EdgeSegment *&part1, EdgeSegment *&part2, EdgeSegment *&part3) const;

};

// A cubic Bezier curve.
class CubicSegment : public EdgeSegment {

public:
	Point2 p[4];

	CubicSegment(Point2 p0, Point2 p1, Point2 p2, Point2 p3, EdgeColor edgeColor = WHITE);
	CubicSegment * clone() const;
	Point2 point(float param) const;
	Vector2 direction(float param) const;
	SignedDistance signedDistance(Point2 origin, float &param) const;
	void bounds(float &l, float &b, float &r, float &t) const;

	void moveStartPoint(Point2 To);
	void moveEndPoint(Point2 to);
	void splitInThirds(EdgeSegment *&part1, EdgeSegment *&part2, EdgeSegment *&part3) const;

};
