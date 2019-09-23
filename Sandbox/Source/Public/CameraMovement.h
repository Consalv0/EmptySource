#pragma once

#include "Components/Component.h"

class CCameraMovement : public EmptySource::CComponent {
	IMPLEMENT_COMPONENT(CCameraMovement)
public:
	EmptySource::Quaternion CameraRotation;
	EmptySource::Quaternion LastCameraRotation;
	EmptySource::Vector2 CursorPosition;
	EmptySource::Vector2 LastCursorPosition;
	float ViewSpeed = 3;

protected:
	typedef EmptySource::CComponent Supper;

	CCameraMovement(EmptySource::GGameObject & GameObject);

	virtual void OnInputEvent(EmptySource::InputEvent & InEvent) override;

	virtual void OnUpdate(const EmptySource::Timestamp & DeltaTime) override;

	virtual void OnDelete() override;
};