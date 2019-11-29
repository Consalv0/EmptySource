
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "Resources/ModelResource.h"
#include "Rendering/MeshPrimitives.h"
#include "Resources/TextureManager.h"
#include "Resources/ShaderManager.h"

#include "../Public/GameStateComponent.h"
#include "../Public/ScenePropComponent.h"

void CSceneProp::ScaleDown(float ScaleFactor) {
	Scale = Math::Clamp01(Scale - (ScaleFactor * ScaleSpeed));
}

CSceneProp::CSceneProp(ESource::GGameObject & GameObject, float ScaleSpeed)
	: CComponent(L"Gun", GameObject), GameStateComponent(NULL), StartingTransform(GameObject.LocalTransform), Scale(1.F), ScaleSpeed(ScaleSpeed) {
}

void CSceneProp::OnUpdate(const ESource::Timestamp & DeltaTime) {
	if (GameStateComponent != NULL && GameStateComponent->GameState == CGameState::EGameState::Starting) {
		GetGameObject().LocalTransform.Scale =
			Math::Mix(StartingTransform.Scale, GetGameObject().LocalTransform.Scale, GameStateComponent->GetAnimationTime());
		Scale = 1.F;
	}
	if (GameStateComponent != NULL && GameStateComponent->GameState == CGameState::EGameState::Started) {
		GetGameObject().LocalTransform.Scale =
			Math::Mix(StartingTransform.Scale * 0.1F, StartingTransform.Scale, Scale);
	}
}

void CSceneProp::OnDelete() { }
