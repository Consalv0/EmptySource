#pragma once

#include "../include/Math/CoreMath.h"

class Transform {
public:
	Vector3 Position;
	Quaternion Rotation;
	Vector3 Scale;

	Transform();

	Vector3 Forward() const;
	Vector3 Up() const;
	Vector3 Right() const;

	Matrix4x4 GetWorldMatrix() const;
};