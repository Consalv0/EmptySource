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

#include "..\include\EdgeSegments.h"
#include "..\include\Math\MathUtility.h"

SignedDistance::SignedDistance() : distance(-MathConstants::Big_Number), dot(1) { }

SignedDistance::SignedDistance(float Distance, float Dot) : distance(Distance), dot(Dot) { }

bool operator<(SignedDistance A, SignedDistance B) {
	return fabs(A.distance) < fabs(B.distance) || (fabs(A.distance) == fabs(B.distance) && A.dot < B.dot);
}

bool operator>(SignedDistance A, SignedDistance B) {
	return fabs(A.distance) > fabs(B.distance) || (fabs(A.distance) == fabs(B.distance) && A.dot > B.dot);
}

bool operator<=(SignedDistance A, SignedDistance B) {
	return fabs(A.distance) < fabs(B.distance) || (fabs(A.distance) == fabs(B.distance) && A.dot <= B.dot);
}

bool operator>=(SignedDistance A, SignedDistance B) {
	return fabs(A.distance) > fabs(B.distance) || (fabs(A.distance) == fabs(B.distance) && A.dot >= B.dot);
}

void EdgeSegment::distanceToPseudoDistance(SignedDistance &Distance, Point2 Origin, float Param) const {
    if (Param < 0) {
        Vector2 Dir = direction(0).Normalized();
        Vector2 aq = Origin - point(0);
        float Dot = Vector2::Dot(aq, Dir);
        if (Dot < 0) {
            float PseudoDistance = Vector2::Cross(aq, Dir);
            if (fabs(PseudoDistance) <= fabs(Distance.distance)) {
                Distance.distance = PseudoDistance;
                Distance.dot = 0;
            }
        }
    } else if (Param > 1) {
        Vector2 Dir = direction(1).Normalized();
        Vector2 bq = Origin - point(1);
        float Dot = Vector2::Dot(bq, Dir);
        if (Dot > 0) {
            float PseudoDistance = Vector2::Cross(bq, Dir);
            if (fabs(PseudoDistance) <= fabs(Distance.distance)) {
                Distance.distance = PseudoDistance;
                Distance.dot = 0;
            }
        }
    }
}

LinearSegment::LinearSegment(Point2 p0, Point2 p1, EdgeColor edgeColor) : EdgeSegment(edgeColor) {
    p[0] = p0;
    p[1] = p1;
}

QuadraticSegment::QuadraticSegment(Point2 p0, Point2 p1, Point2 p2, EdgeColor edgeColor) : EdgeSegment(edgeColor) {
    if (p1 == p0 || p1 == p2)
        p1 = 0.5F * (p0+p2);
    p[0] = p0;
    p[1] = p1;
    p[2] = p2;
}

CubicSegment::CubicSegment(Point2 p0, Point2 p1, Point2 p2, Point2 p3, EdgeColor edgeColor) : EdgeSegment(edgeColor) {
    p[0] = p0;
    p[1] = p1;
    p[2] = p2;
    p[3] = p3;
}

LinearSegment * LinearSegment::clone() const {
    return new LinearSegment(p[0], p[1], color);
}

QuadraticSegment * QuadraticSegment::clone() const {
    return new QuadraticSegment(p[0], p[1], p[2], color);
}

CubicSegment * CubicSegment::clone() const {
    return new CubicSegment(p[0], p[1], p[2], p[3], color);
}

Point2 LinearSegment::point(float param) const {
    return Math::Mix(p[0], p[1], param);
}

Point2 QuadraticSegment::point(float param) const {
    return Math::Mix(Math::Mix(p[0], p[1], param), Math::Mix(p[1], p[2], param), param);
}

Point2 CubicSegment::point(float param) const {
    Vector2 p12 = Math::Mix(p[1], p[2], param);
    return Math::Mix(Math::Mix(Math::Mix(p[0], p[1], param), p12, param), Math::Mix(p12, Math::Mix(p[2], p[3], param), param), param);
}

Vector2 LinearSegment::direction(float param) const {
    return p[1]-p[0];
}

Vector2 QuadraticSegment::direction(float param) const {
    return Math::Mix(p[1]-p[0], p[2]-p[1], param);
}

Vector2 CubicSegment::direction(float param) const {
    Vector2 tangent = Math::Mix(Math::Mix(p[1]-p[0], p[2]-p[1], param), Math::Mix(p[2]-p[1], p[3]-p[2], param), param);
    if (!tangent) {
        if (param == 0) return p[2]-p[0];
        if (param == 1) return p[3]-p[1];
    }
    return tangent;
}

SignedDistance LinearSegment::signedDistance(Point2 origin, float &param) const {
    Vector2 aq = origin-p[0];
    Vector2 ab = p[1]-p[0];
    param = Vector2::Dot(aq, ab)/Vector2::Dot(ab, ab);
    Vector2 eq = p[param > .5]-origin;
    float endpointDistance = eq.Magnitude();
    if (param > 0 && param < 1) {
        float orthoDistance = Vector2::Dot(ab.Orthonormal(false), aq);
        if (fabs(orthoDistance) < endpointDistance)
            return SignedDistance(orthoDistance, 0);
    }
    return SignedDistance(Math::NonZeroSign(Vector2::Cross(aq, ab))*endpointDistance, fabs(Vector2::Dot(ab.Normalized(), eq.Normalized())));
}

SignedDistance QuadraticSegment::signedDistance(Point2 Origin, float &Param) const {
    Vector2 qa = p[0]-Origin;
    Vector2 ab = p[1]-p[0];
    Vector2 br = p[0]+p[2]-p[1]-p[1];
    float a = Vector2::Dot(br, br);
    float b = 3.F * Vector2::Dot(ab, br);
    float c = 2.F * Vector2::Dot(ab, ab)+Vector2::Dot(qa, br);
    float d = Vector2::Dot(qa, ab);
    float t[3];
    int Solutions = MathEquations::SolveCubic(t, a, b, c, d);

    float MinDistance = Math::NonZeroSign(Vector2::Cross(ab, qa))*qa.Magnitude(); // distance from A
    Param = -Vector2::Dot(qa, ab)/Vector2::Dot(ab, ab);
    {
        float distance = Math::NonZeroSign(Vector2::Cross(p[2]-p[1], p[2]-Origin))*(p[2]-Origin).Magnitude(); // distance from B
        if (fabs(distance) < fabs(MinDistance)) {
            MinDistance = distance;
            Param = Vector2::Dot(Origin-p[1], p[2]-p[1])/Vector2::Dot(p[2]-p[1], p[2]-p[1]);
        }
    }
    for (int i = 0; i < Solutions; ++i) {
        if (t[i] > 0 && t[i] < 1) {
            Point2 EndPoint = p[0]+2*t[i]*ab+t[i]*t[i]*br;
            float distance = Math::NonZeroSign(Vector2::Cross(p[2]-p[0], EndPoint-Origin))*(EndPoint-Origin).Magnitude();
            if (fabs(distance) <= fabs(MinDistance)) {
                MinDistance = distance;
                Param = t[i];
            }
        }
    }

    if (Param >= 0 && Param <= 1)
        return SignedDistance(MinDistance, 0);
    if (Param < .5)
        return SignedDistance(MinDistance, fabs(Vector2::Dot(ab.Normalized(), qa.Normalized())));
    else
        return SignedDistance(MinDistance, fabs(Vector2::Dot((p[2]-p[1]).Normalized(), (p[2]-Origin).Normalized())));
}

SignedDistance CubicSegment::signedDistance(Point2 Origin, float &param) const {
    Vector2 qa = p[0] - Origin;
    Vector2 ab = p[1] - p[0];
    Vector2 br = p[2] - p[1]-ab;
    Vector2 as = (p[3] - p[2]) - (p[2] - p[1]) - br;

    Vector2 epDir = direction(0);
	// --- Distance from A
    float minDistance = Math::NonZeroSign(Vector2::Cross(epDir, qa))*qa.Magnitude(); 
    param = -Vector2::Dot(qa, epDir)/Vector2::Dot(epDir, epDir);
    {
        epDir = direction(1);
		// --- Distance from B
        float distance = Math::NonZeroSign(Vector2::Cross(epDir, p[3] - Origin)) * (p[3] - Origin).Magnitude(); 
        if (fabs(distance) < fabs(minDistance)) {
            minDistance = distance;
            param = Vector2::Dot(Origin+epDir-p[3], epDir)/Vector2::Dot(epDir, epDir);
        }
    }
    // Iterative minimum distance search
    for (int i = 0; i <= MSDFGEN_CUBIC_SEARCH_STARTS; ++i) {
        float t = (float) i/MSDFGEN_CUBIC_SEARCH_STARTS;
        for (int step = 0;; ++step) {
            Vector2 qpt = point(t)-Origin;
            float distance = Math::NonZeroSign(Vector2::Cross(direction(t), qpt))*qpt.Magnitude();
            if (fabs(distance) < fabs(minDistance)) {
                minDistance = distance;
                param = t;
            }
            if (step == MSDFGEN_CUBIC_SEARCH_STEPS)
                break;
            // Improve t
            Vector2 d1 = 3*as*t*t+6*br*t+3*ab;
            Vector2 d2 = 6*as*t+6*br;
            t -= Vector2::Dot(qpt, d1)/(Vector2::Dot(d1, d1)+Vector2::Dot(qpt, d2));
            if (t < 0 || t > 1)
                break;
        }
    }

    if (param >= 0 && param <= 1)
        return SignedDistance(minDistance, 0);
    if (param < .5F)
        return SignedDistance(minDistance, fabs(Vector2::Dot(direction(0).Normalized(), qa.Normalized())));
    else
        return SignedDistance(minDistance, fabs(Vector2::Dot(direction(1).Normalized(), (p[3]-Origin).Normalized())));
}

static void pointBounds(Point2 p, float &l, float &b, float &r, float &t) {
    if (p.x < l) l = p.x;
    if (p.y < b) b = p.y;
    if (p.x > r) r = p.x;
    if (p.y > t) t = p.y;
}

void LinearSegment::bounds(float &l, float &b, float &r, float &t) const {
    pointBounds(p[0], l, b, r, t);
    pointBounds(p[1], l, b, r, t);
}

void QuadraticSegment::bounds(float &l, float &b, float &r, float &t) const {
    pointBounds(p[0], l, b, r, t);
    pointBounds(p[2], l, b, r, t);
    Vector2 bot = (p[1]-p[0])-(p[2]-p[1]);
    if (bot.x) {
        float param = (p[1].x-p[0].x)/bot.x;
        if (param > 0 && param < 1)
            pointBounds(point(param), l, b, r, t);
    }
    if (bot.y) {
        float param = (p[1].y-p[0].y)/bot.y;
        if (param > 0 && param < 1)
            pointBounds(point(param), l, b, r, t);
    }
}

void CubicSegment::bounds(float &l, float &b, float &r, float &t) const {
    pointBounds(p[0], l, b, r, t);
    pointBounds(p[3], l, b, r, t);
    Vector2 a0 = p[1]-p[0];
    Vector2 a1 = 2*(p[2]-p[1]-a0);
    Vector2 a2 = p[3]-3*p[2]+3*p[1]-p[0];
    float params[2];
    int solutions;
    solutions = MathEquations::SolveQuadratic(params, a2.x, a1.x, a0.x);
    for (int i = 0; i < solutions; ++i)
        if (params[i] > 0 && params[i] < 1)
            pointBounds(point(params[i]), l, b, r, t);
    solutions = MathEquations::SolveQuadratic(params, a2.y, a1.y, a0.y);
    for (int i = 0; i < solutions; ++i)
        if (params[i] > 0 && params[i] < 1)
            pointBounds(point(params[i]), l, b, r, t);
}

void LinearSegment::moveStartPoint(Point2 To) {
    p[0] = To;
}

void QuadraticSegment::moveStartPoint(Point2 To) {
    Vector2 origSDir = p[0]-p[1];
    Point2 origP1 = p[1];
    p[1] += Vector2::Cross(p[0]-p[1], To-p[0])/Vector2::Cross(p[0]-p[1], p[2]-p[1])*(p[2]-p[1]);
    p[0] = To;
    if (Vector2::Dot(origSDir, p[0]-p[1]) < 0)
        p[1] = origP1;
}

void CubicSegment::moveStartPoint(Point2 To) {
    p[1] += To-p[0];
    p[0] = To;
}

void LinearSegment::moveEndPoint(Point2 To) {
    p[1] = To;
}

void QuadraticSegment::moveEndPoint(Point2 To) {
    Vector2 origEDir = p[2]-p[1];
    Point2 origP1 = p[1];
    p[1] += Vector2::Cross(p[2]-p[1], To-p[2])/Vector2::Cross(p[2]-p[1], p[0]-p[1])*(p[0]-p[1]);
    p[2] = To;
    if (Vector2::Dot(origEDir, p[2]-p[1]) < 0)
        p[1] = origP1;
}

void CubicSegment::moveEndPoint(Point2 To) {
    p[2] += To-p[3];
    p[3] = To;
}

void LinearSegment::splitInThirds(EdgeSegment *&Part1, EdgeSegment *&Part2, EdgeSegment *&Part3) const {
    Part1 = new LinearSegment(p[0], point(1/3.F), color);
    Part2 = new LinearSegment(point(1/3.F), point(2/3.F), color);
    Part3 = new LinearSegment(point(2/3.F), p[1], color);
}

void QuadraticSegment::splitInThirds(EdgeSegment *&Part1, EdgeSegment *&Part2, EdgeSegment *&Part3) const {
	using namespace Math;
    Part1 = new QuadraticSegment(p[0], Mix(p[0], p[1], 1/3.F), point(1/3.F), color);
    Part2 = new QuadraticSegment(point(1/3.F), Mix(Mix(p[0], p[1], 5/9.F), Mix(p[1], p[2], 4/9.F), .5F), point(2/3.F), color);
    Part3 = new QuadraticSegment(point(2/3.F), Mix(p[1], p[2], 2/3.F), p[2], color);
}

void CubicSegment::splitInThirds(EdgeSegment *&Part1, EdgeSegment *&Part2, EdgeSegment *&Part3) const {
	using namespace Math;
    Part1 = new CubicSegment(
		p[0], p[0] == p[1] ? p[0] : Mix(p[0], p[1], 1/3.F),
		Mix(Mix(p[0], p[1], 1/3.F), Mix(p[1], p[2], 1/3.F), 1/3.F),
		point(1/3.F), color
	);
    Part2 = new CubicSegment(
		point(1/3.F),
		Mix(Mix(Mix(p[0], p[1], 1/3.F), Mix(p[1], p[2], 1/3.F), 1/3.F), Mix(Mix(p[1], p[2], 1/3.F), Mix(p[2], p[3], 1/3.F), 1/3.F), 2/3.F),
        Mix(Mix(Mix(p[0], p[1], 2/3.F), Mix(p[1], p[2], 2/3.F), 2/3.F), Mix(Mix(p[1], p[2], 2/3.F), Mix(p[2], p[3], 2/3.F), 2/3.F), 1/3.F),
        point(2/3.F), color
	);
    Part3 = new CubicSegment(
		point(2/3.F), Mix(Mix(p[1], p[2], 2/3.F),
		Mix(p[2], p[3], 2/3.F), 2/3.F), p[2] == p[3] ? p[3] : Mix(p[2], p[3], 2/3.F),
		p[3], color
	);
}