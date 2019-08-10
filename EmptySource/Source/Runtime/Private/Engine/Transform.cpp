
#include "CoreMinimal.h"
#include "Engine/Transform.h"

namespace EmptySource {

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

	Matrix4x4 Transform::GetGLViewMatrix() const {
		Vector3 const Forward(((Position + Rotation * Vector3(0, 0, 1)) - Position).Normalized());
		Vector3 const Side(Forward.Cross(Rotation * Vector3(0, 1)));
		Vector3 const Upper(Side.Cross(Forward));

		return Matrix4x4(
			Side.x, Upper.x, -Forward.x, 0,
			Side.y, Upper.y, -Forward.y, 0,
			Side.z, Upper.z, -Forward.z, 0,
			-Side.Dot(Position), -Upper.Dot(Position), Forward.Dot(Position), 1
		);
		// return Matrix4x4::Scaling(Vector3(1, 1, -1)).Inversed() * Matrix4x4::Rotation(Rotation.Conjugated()) * Matrix4x4::Translation(-Position);
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

}