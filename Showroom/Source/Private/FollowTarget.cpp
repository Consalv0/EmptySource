
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "../Public/FollowTarget.h"

CFollowTarget::CFollowTarget(ESource::GGameObject & GameObject)
	: CComponent(L"FollowTarget", GameObject) {
}

void CFollowTarget::OnUpdate(const ESource::Timestamp & DeltaTime) {
	if (Target != NULL) {
		ESource::Transform & ThisTransform = GetGameObject().LocalTransform;
		ESource::Transform & OtherTransform = Target->LocalTransform;

		ESource::Vector3 TargetPosition = OtherTransform.Position;

		if (ModuleMovement > 0.0F) {
			TargetPosition.X = ESource::Math::Abs(TargetPosition.X - ThisTransform.Position.X) >= ModuleMovement ?
				TargetPosition.X - std::fmodf(TargetPosition.X, ModuleMovement) : ThisTransform.Position.X;
			TargetPosition.Y = ESource::Math::Abs(TargetPosition.Y - ThisTransform.Position.Y) >= ModuleMovement ?
				TargetPosition.Y - std::fmodf(TargetPosition.Y, ModuleMovement) : ThisTransform.Position.Y;
			TargetPosition.Z = ESource::Math::Abs(TargetPosition.Z - ThisTransform.Position.Z) >= ModuleMovement ?
				TargetPosition.Z - std::fmodf(TargetPosition.Z, ModuleMovement) : ThisTransform.Position.Z;
		}

		if (FixedPositionAxisX) TargetPosition.X = ThisTransform.Position.X;
		if (FixedPositionAxisY) TargetPosition.Y = ThisTransform.Position.Y;
		if (FixedPositionAxisZ) TargetPosition.Z = ThisTransform.Position.Z;

		ThisTransform.Position = ESource::Math::Mix(ThisTransform.Position, TargetPosition, DeltaSpeed);
	}
}

void CFollowTarget::OnDelete() { }
