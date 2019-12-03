#pragma once

#include "Components/Component.h"
#include "Components/ComponentAnimable.h"

#include "Components/ComponentCamera.h"
#include "Components/ComponentPhysicBody.h"

#include "../Public/GameStateComponent.h"

class CProp : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CProp)
public:
	ESource::CCamera * PlayerCamera;
	ESource::CCamera * HunterCamera;
	ESource::CPhysicBody * PhysicBody;
	CGameState * GameStateComponent;

	void SetPlayerCamera(ESource::CCamera * PlayerCamera);

	bool bReloading;

protected:
	typedef ESource::CComponent Supper;

	ESource::Transform StartingTransform;

	CProp(ESource::GGameObject & GameObject);

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnDelete() override;

	virtual void OnPostRender() override;

private:
	ESource::Material RenderTextureMaterial;

};