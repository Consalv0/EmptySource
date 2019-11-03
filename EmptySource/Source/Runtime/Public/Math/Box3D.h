#pragma once

#include "Math/MathUtility.h"
#include "Math/Matrix4x4.h"
#include "Math/Vector3.h"

namespace ESource {

	struct Box3D {
	public:
		union {
			struct { float Left, Bottom, Back, Right, Top, Front; };
			struct { float MinX, MinY, MinZ, MaxX, MaxY, MaxZ; };
		};

		Box3D() {
			MinX = MinY = MinZ = MathConstants::BigNumber;
			MaxX = MaxY = MaxZ = -MathConstants::BigNumber;
		}

		Box3D(float MinX, float MinY, float MinZ, float MaxX, float MaxY, float MaxZ)
			: MinX(MinX), MinY(MinY), MinZ(MinZ), MaxX(MaxX), MaxY(MaxY), MaxZ(MaxZ)
		{ }

		inline Box3D Transform(const Matrix4x4 & Transformation) const {
			Box3D Value;
			// Min-Max N
			Value.Add(Transformation.MultiplyPoint(Vector3(MinX, MinY, MinZ)));
			Value.Add(Transformation.MultiplyPoint(Vector3(MaxX, MaxY, MaxZ)));
			// Min-Max X
			Value.Add(Transformation.MultiplyPoint(Vector3(MaxX, MinY, MinZ)));
			Value.Add(Transformation.MultiplyPoint(Vector3(MinX, MaxY, MaxZ)));
			// Min-Max Y
			Value.Add(Transformation.MultiplyPoint(Vector3(MinX, MaxY, MinZ)));
			Value.Add(Transformation.MultiplyPoint(Vector3(MaxX, MinY, MaxZ)));
			// Min-Max Z
			Value.Add(Transformation.MultiplyPoint(Vector3(MinX, MinY, MaxZ)));
			Value.Add(Transformation.MultiplyPoint(Vector3(MaxX, MaxY, MinZ)));
			return Value;
		}

		//* Add point to the BondingBox
		inline void Add(Point3 Point) {
			MinX = Math::Min(MinX, Point.X); MinY = Math::Min(MinY, Point.Y); MinZ = Math::Min(MinZ, Point.Z);
			MaxX = Math::Max(MaxX, Point.X); MaxY = Math::Max(MaxY, Point.Y); MaxZ = Math::Max(MaxZ, Point.Z);
		};

		//* Get the dimensions of the bounding box
		inline Vector3 GetSize() const { return Vector3(GetWidth(), GetHeight(), GetDepth()); }

		//* Get the center position of the bounding box
		inline Vector3 GetCenter() const { return Vector3(MinX + MaxX, MinY + MaxY, MinZ + MaxZ) * .5F; }

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
		
		// Used in frustrum computations
		inline Point3 GetPointPositive(const Vector3 & Normal) const {
			Vector3 MaxPoint = GetMaxPoint();
			Point3 Result = GetMinPoint();
			if (Normal.X > 0) Result.X = MaxPoint.X;
			if (Normal.Y > 0) Result.Y = MaxPoint.Y;
			if (Normal.Z > 0) Result.Z = MaxPoint.Z;
			return Result;
		}

		// Used in frustrum computations
		inline Point3 GetPointNegative(const Vector3 & Normal) const {
			Vector3 MaxPoint = GetMaxPoint();
			Point3 Result = GetMinPoint();
			if (Normal.X < 0) Result.X = MaxPoint.X;
			if (Normal.Y < 0) Result.Y = MaxPoint.Y;
			if (Normal.Z < 0) Result.Z = MaxPoint.Z;
			return Result;
		}

	};

	typedef Box3D BoundingBox3D;

}