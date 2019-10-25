
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
			TargetPosition.x = ESource::Math::Abs(TargetPosition.x - ThisTransform.Position.x) >= ModuleMovement ?
				TargetPosition.x - std::fmodf(TargetPosition.x, ModuleMovement) : ThisTransform.Position.x;
			TargetPosition.y = ESource::Math::Abs(TargetPosition.y - ThisTransform.Position.y) >= ModuleMovement ?
				TargetPosition.y - std::fmodf(TargetPosition.y, ModuleMovement) : ThisTransform.Position.y;
			TargetPosition.z = ESource::Math::Abs(TargetPosition.z - ThisTransform.Position.z) >= ModuleMovement ?
				TargetPosition.z - std::fmodf(TargetPosition.z, ModuleMovement) : ThisTransform.Position.z;
		}

		if (FixedPositionAxisX) TargetPosition.x = ThisTransform.Position.x;
		if (FixedPositionAxisY) TargetPosition.y = ThisTransform.Position.y;
		if (FixedPositionAxisZ) TargetPosition.z = ThisTransform.Position.z;

		ThisTransform.Position = ESource::Math::Mix(ThisTransform.Position, TargetPosition, DeltaSpeed);
	}
}

void CFollowTarget::OnDelete() { }
