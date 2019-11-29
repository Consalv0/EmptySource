#pragma once

#include "Components/Component.h"

class CCameraMovement : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CCameraMovement)
public:
	class CGameState * GameStateComponent;
	ESource::Quaternion CameraRotation;
	ESource::Quaternion LastCameraRotation;
	ESource::Vector2 CursorPosition;
	ESource::Vector2 LastCursorPosition;
	float ViewSpeed = 5.5F;
	float UpVelocity = 0.F;
	float DefaultHeight = 0.F;
	int InputIndex = -1;

protected:
	typedef ESource::CComponent Supper;

	CCameraMovement(ESource::GGameObject & GameObject);

	virtual void OnInputEvent(ESource::InputEvent & InEvent) override;

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnDelete() override;

private:
	float JumpForce = 0.F;
};