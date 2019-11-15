
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "Resources/ModelResource.h"
#include "Physics/PhysicsWorld.h"
#include "Physics/Ray.h"

#include "../Public/PropComponent.h"
#include "Components/ComponentPhysicBody.h"

CProp::CProp(ESource::GGameObject & GameObject)
	: CComponent(L"Gun", GameObject) {
}

void CProp::OnUpdate(const ESource::Timestamp & DeltaTime) {
}

void CProp::OnDelete() { }
