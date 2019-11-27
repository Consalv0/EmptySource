#pragma once

#include "Components/Component.h"
#include "Components/ComponentAnimable.h"

#include "Components/ComponentCamera.h"

class CGun : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CGun)
public:
	void SetGunObjects(ESource::GGameObject * GunObject, ESource::CAnimable * GunAnimable, ESource::CCamera * PlayerCamera);

	bool bReloading;

protected:
	typedef ESource::CComponent Supper;

	ESource::Material RenderTextureMaterial;
	ESource::CCamera * PlayerCamera;
	ESource::GGameObject * GunObject;
	ESource::CAnimable * GunAnimable;

	CGun(ESource::GGameObject & GameObject);

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnPostRender() override;

	virtual void OnDelete() override;
};