#include "../include/Transform.h"

Transform::Transform() : Position(), Rotation(), Scale(1.F) {
}

Transform::Transform(const Point3 & Position, const Quaternion & Rotation, const Vector3 & Scale) :
	Position(Position), Rotation(Rotation), Scale(Scale) {
}

Transform::Transform(const Point3 & Position, const Quaternion & Rotation) :
	Position(Position), Rotation(Rotation), Scale(1.F) {
}

Transform::Transform(const Point3 & Position) :
	Position(Position), Rotation(), Scale(1.F) {
}

Matrix4x4 Transform::GetWorldToLocalMatrix() const {
	return GetLocalToWorldMatrix().Inversed();
}

Matrix4x4 Transform::GetLocalToWorldMatrix() const {
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
