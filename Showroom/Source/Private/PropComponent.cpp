
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "Resources/ModelResource.h"
#include "Physics/PhysicsWorld.h"
#include "Physics/Ray.h"

#include "../Public/PropComponent.h"
#include "../Public/ScenePropComponent.h"

#include "Components/ComponentPhysicBody.h"

CProp::CProp(ESource::GGameObject & GameObject)
	: CComponent(L"Gun", GameObject), GameStateComponent(NULL), PhysicBody(NULL) {
}

void CProp::SetPlayerCamera(ESource::CCamera * InPlayerCamera) {
	PlayerCamera = InPlayerCamera;
	StartingTransform = PlayerCamera->GetGameObject().GetParent()->LocalTransform;
}

void CProp::OnUpdate(const ESource::Timestamp & DeltaTime) {
	if (GameStateComponent != NULL && GameStateComponent->GameState == CGameState::EGameState::Started) {
		TArray<ESource::CPhysicBody *> Intersections;
		ESource::Application::GetInstance()->GetPhysicsWorld().AABBIntersection(
			PhysicBody->GetMeshData()->Bounding.Transform(GetGameObject().GetWorldMatrix()), Intersections
		);

		for (auto & Intersection : Intersections) {
			auto SceneProp = Intersection->GetGameObject().GetFirstComponent<CSceneProp>();
			if (SceneProp != NULL) {
				SceneProp->ScaleDown(DeltaTime.GetDeltaTime<ESource::Time::Second>() * 0.1F);
			}
		}
	}
	if (GameStateComponent != NULL && GameStateComponent->GameState == CGameState::EGameState::Starting) {
		PlayerCamera->GetGameObject().GetParent()->LocalTransform.Position =
			Math::Mix(StartingTransform.Position, PlayerCamera->GetGameObject().GetParent()->LocalTransform.Position, GameStateComponent->GetAnimationTime());
	}
}

void CProp::OnDelete() { }
