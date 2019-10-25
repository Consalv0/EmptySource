#pragma once

#include "Components/Component.h"

class CFollowTarget : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CFollowTarget)
public:
	float DeltaSpeed = 1.F;

	ESource::GGameObject * Target;

	float ModuleMovement = 0.0F;

	bool FixedPositionAxisX = false;
	bool FixedPositionAxisY = false;
	bool FixedPositionAxisZ = false;

protected:
	typedef ESource::CComponent Supper;

	CFollowTarget(ESource::GGameObject & GameObject);

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnDelete() override;
};