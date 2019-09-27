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

#include "CoreMinimal.h"
#include "Fonts/EdgeSegments.h"
#include "Math/MathUtility.h"

namespace ESource {

	SignedDistance::SignedDistance() : Distance(-MathConstants::BigNumber), Dot(1) { }

	SignedDistance::SignedDistance(float Distance, float Dot) : Distance(Distance), Dot(Dot) { }

	bool operator<(SignedDistance A, SignedDistance B) {
		return fabs(A.Distance) < fabs(B.Distance) || (fabs(A.Distance) == fabs(B.Distance) && A.Dot < B.Dot);
	}

	bool operator>(SignedDistance A, SignedDistance B) {
		return fabs(A.Distance) > fabs(B.Distance) || (fabs(A.Distance) == fabs(B.Distance) && A.Dot > B.Dot);
	}

	bool operator<=(SignedDistance A, SignedDistance B) {
		return fabs(A.Distance) < fabs(B.Distance) || (fabs(A.Distance) == fabs(B.Distance) && A.Dot <= B.Dot);
	}

	bool operator>=(SignedDistance A, SignedDistance B) {
		return fabs(A.Distance) > fabs(B.Distance) || (fabs(A.Distance) == fabs(B.Distance) && A.Dot >= B.Dot);
	}

	void EdgeSegment::DistanceToPseudoDistance(SignedDistance &Distance, Point2 Origin, float Param) const {
		if (Param < 0) {
			Vector2 Dir = DirectionAt(0).Normalized();
			Vector2 aq = Origin - PointAt(0);
			float Dot = Vector2::Dot(aq, Dir);
			if (Dot < 0) {
				float PseudoDistance = Vector2::Cross(aq, Dir);
				if (fabs(PseudoDistance) <= fabs(Distance.Distance)) {
					Distance.Distance = PseudoDistance;
					Distance.Dot = 0;
				}
			}
		}
		else if (Param > 1) {
			Vector2 Dir = DirectionAt(1).Normalized();
			Vector2 bq = Origin - PointAt(1);
			float Dot = Vector2::Dot(bq, Dir);
			if (Dot > 0) {
				float PseudoDistance = Vector2::Cross(bq, Dir);
				if (fabs(PseudoDistance) <= fabs(Distance.Distance)) {
					Distance.Distance = PseudoDistance;
					Distance.Dot = 0;
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
			p1 = 0.5F * (p0 + p2);
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

	LinearSegment * LinearSegment::Clone() const {
		return new LinearSegment(p[0], p[1], Color);
	}

	QuadraticSegment * QuadraticSegment::Clone() const {
		return new QuadraticSegment(p[0], p[1], p[2], Color);
	}

	CubicSegment * CubicSegment::Clone() const {
		return new CubicSegment(p[0], p[1], p[2], p[3], Color);
	}

	Point2 LinearSegment::PointAt(float Value) const {
		return Math::Mix(p[0], p[1], Value);
	}

	Point2 QuadraticSegment::PointAt(float Value) const {
		return Math::Mix(Math::Mix(p[0], p[1], Value), Math::Mix(p[1], p[2], Value), Value);
	}

	Point2 CubicSegment::PointAt(float Value) const {
		Vector2 p12 = Math::Mix(p[1], p[2], Value);
		return Math::Mix(Math::Mix(Math::Mix(p[0], p[1], Value), p12, Value), Math::Mix(p12, Math::Mix(p[2], p[3], Value), Value), Value);
	}

	Vector2 LinearSegment::DirectionAt(float Value) const {
		return p[1] - p[0];
	}

	Vector2 QuadraticSegment::DirectionAt(float Value) const {
		return Math::Mix(p[1] - p[0], p[2] - p[1], Value);
	}

	Vector2 CubicSegment::DirectionAt(float Value) const {
		Vector2 Tangent = Math::Mix(Math::Mix(p[1] - p[0], p[2] - p[1], Value), Math::Mix(p[2] - p[1], p[3] - p[2], Value), Value);
		if (!Tangent) {
			if (Value == 0) return p[2] - p[0];
			if (Value == 1) return p[3] - p[1];
		}
		return Tangent;
	}

	SignedDistance LinearSegment::GetSignedDistance(Point2 Origin, float &Value) const {
		Vector2 aq = Origin - p[0];
		Vector2 ab = p[1] - p[0];
		Value = Vector2::Dot(aq, ab) / Vector2::Dot(ab, ab);
		Vector2 eq = p[Value > .5F] - Origin;
		float EndPointDistance = eq.Magnitude();
		if (Value > 0.F && Value < 1.F) {
			float OrthoDistance = Vector2::Dot(ab.Orthonormal(false), aq);
			if (fabs(OrthoDistance) < EndPointDistance)
				return SignedDistance(OrthoDistance, 0.F);
		}
		return SignedDistance(
			Math::NonZeroSign(Vector2::Cross(aq, ab)) * EndPointDistance,
			fabs(Vector2::Dot(ab.Normalized(), eq.Normalized())
			));
	}

	SignedDistance QuadraticSegment::GetSignedDistance(Point2 Origin, float &Param) const {
		Vector2 qa = p[0] - Origin;
		Vector2 ab = p[1] - p[0];
		Vector2 br = p[0] + p[2] - p[1] - p[1];
		float a = Vector2::Dot(br, br);
		float b = 3.F * Vector2::Dot(ab, br);
		float c = 2.F * Vector2::Dot(ab, ab) + Vector2::Dot(qa, br);
		float d = Vector2::Dot(qa, ab);
		float t[3];
		int Solutions = MathEquations::SolveCubic(t, a, b, c, d);

		// --- Distance from A
		float MinDistance = Math::NonZeroSign(Vector2::Cross(ab, qa))*qa.Magnitude();
		Param = -Vector2::Dot(qa, ab) / Vector2::Dot(ab, ab);
		{
			// --- Distance from B
			float Distance = Math::NonZeroSign(Vector2::Cross(p[2] - p[1], p[2] - Origin))*(p[2] - Origin).Magnitude();
			if (fabs(Distance) < fabs(MinDistance)) {
				MinDistance = Distance;
				Param = Vector2::Dot(Origin - p[1], p[2] - p[1]) / Vector2::Dot(p[2] - p[1], p[2] - p[1]);
			}
		}
		for (int i = 0; i < Solutions; ++i) {
			if (t[i] > 0 && t[i] < 1) {
				Point2 EndPoint = p[0] + 2 * t[i] * ab + t[i] * t[i] * br;
				float Distance = Math::NonZeroSign(Vector2::Cross(p[2] - p[0], EndPoint - Origin))*(EndPoint - Origin).Magnitude();
				if (fabs(Distance) <= fabs(MinDistance)) {
					MinDistance = Distance;
					Param = t[i];
				}
			}
		}

		if (Param >= 0 && Param <= 1)
			return SignedDistance(MinDistance, 0);
		if (Param < .5F)
			return SignedDistance(MinDistance, fabs(Vector2::Dot(ab.Normalized(), qa.Normalized())));
		else
			return SignedDistance(MinDistance, fabs(Vector2::Dot((p[2] - p[1]).Normalized(), (p[2] - Origin).Normalized())));
	}

	SignedDistance CubicSegment::GetSignedDistance(Point2 Origin, float &Param) const {
		Vector2 qa = p[0] - Origin;
		Vector2 ab = p[1] - p[0];
		Vector2 br = p[2] - p[1] - ab;
		Vector2 as = (p[3] - p[2]) - (p[2] - p[1]) - br;

		Vector2 epDir = DirectionAt(0);
		// --- Distance from A
		float MinDistance = Math::NonZeroSign(Vector2::Cross(epDir, qa)) * qa.Magnitude();
		Param = -Vector2::Dot(qa, epDir) / Vector2::Dot(epDir, epDir);
		{
			epDir = DirectionAt(1);
			// --- Distance from B
			float Distance = Math::NonZeroSign(Vector2::Cross(epDir, p[3] - Origin)) * (p[3] - Origin).Magnitude();
			if (fabs(Distance) < fabs(MinDistance)) {
				MinDistance = Distance;
				Param = Vector2::Dot(Origin + epDir - p[3], epDir) / Vector2::Dot(epDir, epDir);
			}
		}
		// --- Iterative minimum distance search
		for (int i = 0; i <= MSDFGEN_CUBIC_SEARCH_STARTS; ++i) {
			float t = (float)i / MSDFGEN_CUBIC_SEARCH_STARTS;
			for (int step = 0;; ++step) {
				Vector2 qpt = PointAt(t) - Origin;
				float Distance = Math::NonZeroSign(Vector2::Cross(DirectionAt(t), qpt))*qpt.Magnitude();
				if (fabs(Distance) < fabs(MinDistance)) {
					MinDistance = Distance;
					Param = t;
				}
				if (step == MSDFGEN_CUBIC_SEARCH_STEPS)
					break;
				// --- Improve t
				Vector2 d1 = 3.F*as*t*t + 6.F*br*t + 3.F*ab;
				Vector2 d2 = 6.F*as*t + 6.F*br;
				t -= Vector2::Dot(qpt, d1) / (Vector2::Dot(d1, d1) + Vector2::Dot(qpt, d2));
				if (t < 0 || t > 1)
					break;
			}
		}

		if (Param >= 0 && Param <= 1)
			return SignedDistance(MinDistance, 0);
		if (Param < .5F)
			return SignedDistance(MinDistance, fabs(Vector2::Dot(DirectionAt(0).Normalized(), qa.Normalized())));
		else
			return SignedDistance(MinDistance, fabs(Vector2::Dot(DirectionAt(1).Normalized(), (p[3] - Origin).Normalized())));
	}

	static void PointBounds(Point2 p, float &l, float &b, float &r, float &t) {
		if (p.x < l) l = p.x;
		if (p.y < b) b = p.y;
		if (p.x > r) r = p.x;
		if (p.y > t) t = p.y;
	}

	void LinearSegment::GetBounds(float &l, float &b, float &r, float &t) const {
		PointBounds(p[0], l, b, r, t);
		PointBounds(p[1], l, b, r, t);
	}

	void QuadraticSegment::GetBounds(float &l, float &b, float &r, float &t) const {
		PointBounds(p[0], l, b, r, t);
		PointBounds(p[2], l, b, r, t);
		Vector2 Bottom = (p[1] - p[0]) - (p[2] - p[1]);

		if (Bottom.x) {
			float Param = (p[1].x - p[0].x) / Bottom.x;
			if (Param > 0 && Param < 1)
				PointBounds(PointAt(Param), l, b, r, t);
		}
		if (Bottom.y) {
			float Param = (p[1].y - p[0].y) / Bottom.y;
			if (Param > 0 && Param < 1)
				PointBounds(PointAt(Param), l, b, r, t);
		}
	}

	void CubicSegment::GetBounds(float &l, float &b, float &r, float &t) const {
		PointBounds(p[0], l, b, r, t);
		PointBounds(p[3], l, b, r, t);
		Vector2 a0 = p[1] - p[0];
		Vector2 a1 = 2.F*(p[2] - p[1] - a0);
		Vector2 a2 = p[3] - 3.F*p[2] + 3.F*p[1] - p[0];
		float Values[2];
		int Solutions;
		Solutions = MathEquations::SolveQuadratic(Values, a2.x, a1.x, a0.x);
		for (int i = 0; i < Solutions; ++i)
			if (Values[i] > 0 && Values[i] < 1)
				PointBounds(PointAt(Values[i]), l, b, r, t);
		Solutions = MathEquations::SolveQuadratic(Values, a2.y, a1.y, a0.y);
		for (int i = 0; i < Solutions; ++i)
			if (Values[i] > 0 && Values[i] < 1)
				PointBounds(PointAt(Values[i]), l, b, r, t);
	}

	void LinearSegment::MoveStartPoint(Point2 To) {
		p[0] = To;
	}

	void QuadraticSegment::MoveStartPoint(Point2 To) {
		Vector2 origSDir = p[0] - p[1];
		Point2 origP1 = p[1];
		p[1] += Vector2::Cross(p[0] - p[1], To - p[0]) / Vector2::Cross(p[0] - p[1], p[2] - p[1])*(p[2] - p[1]);
		p[0] = To;
		if (Vector2::Dot(origSDir, p[0] - p[1]) < 0)
			p[1] = origP1;
	}

	void CubicSegment::MoveStartPoint(Point2 To) {
		p[1] += To - p[0];
		p[0] = To;
	}

	void LinearSegment::MoveEndPoint(Point2 To) {
		p[1] = To;
	}

	void QuadraticSegment::MoveEndPoint(Point2 To) {
		Vector2 origEDir = p[2] - p[1];
		Point2 origP1 = p[1];
		p[1] += Vector2::Cross(p[2] - p[1], To - p[2]) / Vector2::Cross(p[2] - p[1], p[0] - p[1])*(p[0] - p[1]);
		p[2] = To;
		if (Vector2::Dot(origEDir, p[2] - p[1]) < 0)
			p[1] = origP1;
	}

	void CubicSegment::MoveEndPoint(Point2 To) {
		p[2] += To - p[3];
		p[3] = To;
	}

	void LinearSegment::SplitInThirds(EdgeSegment *&Part1, EdgeSegment *&Part2, EdgeSegment *&Part3) const {
		Part1 = new LinearSegment(p[0], PointAt(1 / 3.F), Color);
		Part2 = new LinearSegment(PointAt(1 / 3.F), PointAt(2 / 3.F), Color);
		Part3 = new LinearSegment(PointAt(2 / 3.F), p[1], Color);
	}

	void QuadraticSegment::SplitInThirds(EdgeSegment *&Part1, EdgeSegment *&Part2, EdgeSegment *&Part3) const {
		using namespace Math;
		Part1 = new QuadraticSegment(p[0], Mix(p[0], p[1], 1 / 3.F), PointAt(1 / 3.F), Color);
		Part2 = new QuadraticSegment(PointAt(1 / 3.F), Mix(Mix(p[0], p[1], 5 / 9.F), Mix(p[1], p[2], 4 / 9.F), .5F), PointAt(2 / 3.F), Color);
		Part3 = new QuadraticSegment(PointAt(2 / 3.F), Mix(p[1], p[2], 2 / 3.F), p[2], Color);
	}

	void CubicSegment::SplitInThirds(EdgeSegment *&Part1, EdgeSegment *&Part2, EdgeSegment *&Part3) const {
		using namespace Math;
		Part1 = new CubicSegment(
			p[0], p[0] == p[1] ? p[0] : Mix(p[0], p[1], 1 / 3.F),
			Mix(Mix(p[0], p[1], 1 / 3.F), Mix(p[1], p[2], 1 / 3.F), 1 / 3.F),
			PointAt(1 / 3.F), Color
		);
		Part2 = new CubicSegment(
			PointAt(1 / 3.F),
			Mix(Mix(Mix(p[0], p[1], 1 / 3.F), Mix(p[1], p[2], 1 / 3.F), 1 / 3.F), Mix(Mix(p[1], p[2], 1 / 3.F), Mix(p[2], p[3], 1 / 3.F), 1 / 3.F), 2 / 3.F),
			Mix(Mix(Mix(p[0], p[1], 2 / 3.F), Mix(p[1], p[2], 2 / 3.F), 2 / 3.F), Mix(Mix(p[1], p[2], 2 / 3.F), Mix(p[2], p[3], 2 / 3.F), 2 / 3.F), 1 / 3.F),
			PointAt(2 / 3.F), Color
		);
		Part3 = new CubicSegment(
			PointAt(2 / 3.F), Mix(Mix(p[1], p[2], 2 / 3.F),
				Mix(p[2], p[3], 2 / 3.F), 2 / 3.F), p[2] == p[3] ? p[3] : Mix(p[2], p[3], 2 / 3.F),
			p[3], Color
		);
	}

}