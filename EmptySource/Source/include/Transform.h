#pragma once

#include "../include/Math/MathUtility.h"
#include "../include/Math/Vector3.h"
#include "../include/Math/Vector4.h"
#include "../include/Math/Quaternion.h"
#include "../include/Math/Matrix4x4.h"

namespace EmptySource {

	class Transform {
	public:
		Point3 Position;
		Quaternion Rotation;
		Vector3 Scale;

		Transform();
		Transform(const Point3 & Position, const Quaternion & Rotation, const Vector3 & Scale);
		Transform(const Point3 & Position, const Quaternion & Rotation);
		Transform(const Point3 & Position);

		Vector3 Forward() const;
		Vector3 Up() const;
		Vector3 Right() const;

		//* Get the inverse of the Model matrix
		Matrix4x4 GetWorldToLocalMatrix() const;

		//* Get the Model matrix
		Matrix4x4 GetLocalToWorldMatrix() const;

		//* Get the Model matrix
		Matrix4x4 GetGLViewMatrix() const;
	};

}