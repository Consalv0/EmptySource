#include "../include/Transform.h"

Transform::Transform() : Position(), Rotation(), Scale(1) {
}

Matrix4x4 Transform::GetWorldMatrix() const {
	return Matrix4x4::Translation(Position) * Rotation.ToMatrix4x4() * Matrix4x4::Scaling(Scale);
}

Vector3 Transform::Forward() const {
	return Rotation * Vector3(0, 0, 1);
}

Vector3 Transform::Up() const {
	return Rotation * Vector3(0, 1);
}

Vector3 Transform::Right() const {
	return Rotation * Vector3(1, 0);
}
