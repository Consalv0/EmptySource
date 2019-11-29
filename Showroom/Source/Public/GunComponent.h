#pragma once

#include "Components/Component.h"
#include "Components/ComponentAnimable.h"

#include "Components/ComponentCamera.h"

#include "../Public/GameStateComponent.h"

class CGun : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CGun)
public:
	CGameState * GameStateComponent;

	bool bReloading;

	void SetGunObjects(ESource::GGameObject * GunObject, ESource::CAnimable * GunAnimable, ESource::CCamera * PlayerCamera);

protected:
	typedef ESource::CComponent Supper;

	ESource::Material RenderTextureMaterial;
	ESource::CCamera * PlayerCamera;
	ESource::GGameObject * GunObject;
	ESource::CAnimable * GunAnimable;
	ESource::Transform StartingTransform;

	CGun(ESource::GGameObject & GameObject);

	void SetPlayerCamera(ESource::CCamera * PlayerCamera);

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnPostRender() override;

	virtual void OnDelete() override;

private:
	int BulletCount;

	void ReduceBullets();

};