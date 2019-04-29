#pragma once

#include "MathUtility.h"
#include "Matrix4x4.h"
#include "Vector3.h"

struct Box3D {
public:
	union {
		struct { float Left, Bottom, Back, Right, Top, Front; };
		struct { float xMin, yMin, zMin, xMax, yMax, zMax; };
	};

	Box3D() {
		xMin = yMin = zMin = MathConstants::BigNumber;
		xMax = yMax = zMax = -MathConstants::BigNumber;
	}

	Box3D(float xMin, float yMin, float zMin, float xMax, float yMax, float zMax) 
	 : xMin(xMin), yMin(yMin), zMin(zMin), xMax(xMax), yMax(yMax), zMax(zMax)
	{ }

	inline Box3D Transform(Matrix4x4 Transformation) const {
		Vector3 MinN = Transformation * (Vector3(xMin, yMin, zMin));
		Vector3 MaxN = Transformation * (Vector3(xMax, yMax, zMax));
		Vector3 MinZ = Transformation * (Vector3(xMin, yMin, zMax));
		Vector3 MaxZ = Transformation * (Vector3(xMax, yMax, zMin));
		Vector3 MinY = Transformation * (Vector3(xMin, yMax, zMin));
		Vector3 MaxY = Transformation * (Vector3(xMax, yMin, zMax));
		Vector3 MinX = Transformation * (Vector3(xMax, yMin, zMin));
		Vector3 MaxX = Transformation * (Vector3(xMin, yMax, zMax));

		Box3D Value;
		Value.Add(MinN); Value.Add(MaxN);
		Value.Add(MinX); Value.Add(MaxX);
		Value.Add(MinZ); Value.Add(MaxZ);
		Value.Add(MinY); Value.Add(MaxY);
		return Value;
	}

	//* Add point to the BondingBox
	inline void Add(Point3 Point) {
		xMin = Math::Min(xMin, Point.x); yMin = Math::Min(yMin, Point.y); zMin = Math::Min(zMin, Point.z);
		xMax = Math::Max(xMax, Point.x); yMax = Math::Max(yMax, Point.y); zMax = Math::Max(zMax, Point.z);
	};

	//* Get the dimensions of the bounding box
	inline Vector3 GetSize() const { return Vector3(GetWidth(), GetHeight(), GetDepth()); }

	//* Get the center position of the bounding box
	inline Vector3 GetCenter() const { return Vector3(xMin + xMax, yMin + yMax, zMin + zMax) * .5F; }

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