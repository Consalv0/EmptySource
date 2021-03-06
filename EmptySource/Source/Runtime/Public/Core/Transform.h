#pragma once

#include "Math/MathUtility.h"
#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/Quaternion.h"
#include "Math/Matrix4x4.h"

namespace ESource {

	class Transform {
	public:
		Point3 Position;
		Quaternion Rotation;
		Vector3 Scale;

		Transform();
		Transform(const Point3 & Position, const Quaternion & Rotation, const Vector3 & Scale);
		Transform(const Point3 & Position, const Quaternion & Rotation);
		Transform(const Point3 & Position);
		Transform(const Matrix4x4 & Matrix);

		Vector3 Forward() const;
		Vector3 Up() const;
		Vector3 Right() const;

		Transform operator*(const Transform & Other) const;

		//* Get the inverse of the Model matrix
		Matrix4x4 GetWorldToLocalMatrix() const;

		//* Get the Model matrix
		Matrix4x4 GetLocalToWorldMatrix() const;

		//* Get the Model matrix
		Matrix4x4 GetGLViewMatrix() const;
	};

}