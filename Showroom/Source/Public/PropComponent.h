#pragma once

#include "Components/Component.h"
#include "Components/ComponentAnimable.h"

#include "Components/ComponentCamera.h"

class CProp : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CProp)
public:
	void SetGunObjects(ESource::GGameObject * GunObject, ESource::CAnimable * GunAnimable, ESource::CCamera * PlayerCamera);

	bool bReloading;

protected:
	typedef ESource::CComponent Supper;

	CProp(ESource::GGameObject & GameObject);

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnDelete() override;
};