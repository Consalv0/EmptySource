#pragma once

#include "MathUtility.h"
#include "Vector3.h"

struct Box3D {
public:
	union {
		struct { float Left, Bottom, Back, Right, Top, Front; };
		struct { float xMin, yMin, zMin, xMax, yMax, zMax; };
	};

	//* Get the lower point of the bounding box
	inline Point3 GetMinPoint() const { return { Math::Min(Left, Right), Math::Min(Top, Bottom), Math::Min(Front, Back) }; }

	//* Get the upper point of the bounding box
	inline Point3 GetMaxPoint() const { return { Math::Max(Left, Right), Math::Max(Top, Bottom), Math::Max(Front, Back) }; }

	//* Get the width of the bounding box
	inline float GetWidth() const { return Math::Max(Left, Right) - Math::Min(Left, Right); }

	//* Get the height of the bounding box
	inline float GetHeight() const { return Math::Max(Top, Bottom) - Math::Min(Top, Bottom); }

	//* Get the depth of the bounding box
	inline float GetDepth() const { return Math::Max(Front, Back) - Math::Min(Front, Back); }

	//* Get the area of the bounding box
	inline float GetArea() const { return GetWidth() * GetHeight() * GetDepth(); }

	//* Get the perimeter of the bounding box
	inline float GetPerimeter() const { return GetWidth() * 2.F + GetHeight() * 2.F + GetDepth() * 2.F; }
};

typedef Box3D BoundingBox3D;