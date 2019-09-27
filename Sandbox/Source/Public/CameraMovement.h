#pragma once

#include "Components/Component.h"

class CCameraMovement : public ESource::CComponent {
	IMPLEMENT_COMPONENT(CCameraMovement)
public:
	ESource::Quaternion CameraRotation;
	ESource::Quaternion LastCameraRotation;
	ESource::Vector2 CursorPosition;
	ESource::Vector2 LastCursorPosition;
	float ViewSpeed = 3;

protected:
	typedef ESource::CComponent Supper;

	CCameraMovement(ESource::GGameObject & GameObject);

	virtual void OnInputEvent(ESource::InputEvent & InEvent) override;

	virtual void OnUpdate(const ESource::Timestamp & DeltaTime) override;

	virtual void OnDelete() override;
};