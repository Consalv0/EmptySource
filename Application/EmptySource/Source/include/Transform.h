#pragma once

#include "../include/Math/MathUtility.h"
#include "../include/Math/Vector3.h"
#include "../include/Math/Vector4.h"
#include "../include/Math/Quaternion.h"
#include "../include/Math/Matrix4x4.h"

class Transform {
public:
	Vector3 Position;
	Quaternion Rotation;
	Vector3 Scale;

	Transform();

	Vector3 Forward() const;
	Vector3 Up() const;
	Vector3 Right() const;

	//* Get the inverse of the Model matrix
	Matrix4x4 GetWorldToLocalMatrix() const;
	
	//* Get the Model matrix
	Matrix4x4 GetLocalToWorldMatrix() const;
};