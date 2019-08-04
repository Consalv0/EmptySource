#pragma once

#include "Math/MathUtility.h"
#include "Math/Vector2.h"

namespace EmptySource {

	struct Box2D {
	public:
		union {
			struct { float Left, Bottom, Right, Top; };
			struct { float xMin, yMin, xMax, yMax; };
		};

		Box2D() = default;

		Box2D(const float & xMin, const float & yMin, const float & xMax, const float & yMax)
			: xMin(xMin), yMin(yMin), xMax(xMax), yMax(yMax)
		{ }

		//* Get the lower point of the bounding box
		inline Point2 GetMinPoint() const { return { Math::Min(Left, Right), Math::Min(Top, Bottom) }; }

		//* Get the upper point of the bounding box
		inline Point2 GetMaxPoint() const { return { Math::Max(Left, Right), Math::Max(Top, Bottom) }; }

		//* Get the width of the bounding box
		inline float GetWidth() const { return Math::Max(Left, Right) - Math::Min(Left, Right); }

		//* Get the height of the bounding box
		inline float GetHeight() const { return Math::Max(Top, Bottom) - Math::Min(Top, Bottom); }

		//* Get the area of the bounding box
		inline float GetArea() const { return GetWidth() * GetHeight(); }

		//* Get the perimeter of the bounding box
		inline float GetPerimeter() const { return GetWidth() * 2.F + GetHeight() * 2.F; }
	};

	typedef Box2D BoundingBox2D;

}