
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "../Public/CameraMovement.h"

using namespace ESource;

CCameraMovement::CCameraMovement(ESource::GGameObject & GameObject)
	: CComponent(L"CameraMovement", GameObject) {
}

void CCameraMovement::OnInputEvent(ESource::InputEvent & InEvent) {
	EventDispatcher<InputEvent> Dispatcher(InEvent);
	Dispatcher.Dispatch<MouseMovedEvent>([this](MouseMovedEvent & Event) {
		CursorPosition = Event.GetMousePosition();
	});
	Dispatcher.Dispatch<MouseButtonPressedEvent>([this](MouseButtonPressedEvent & Event) {
		if (Event.GetMouseButton() == 3 && Event.GetRepeatCount() <= 0) {
			LastCursorPosition = { Input::GetMouseX(), Input::GetMouseY() };
			LastCameraRotation = CameraRotation;
		}
	});
}

void CCameraMovement::OnUpdate(const ESource::Timestamp & DeltaTime) {
	if (Input::IsMouseButtonDown(3)) {
		Vector3 EulerAngles = LastCameraRotation.ToEulerAngles();
		CameraRotation = Quaternion::EulerAngles(EulerAngles + Vector3(Input::GetMouseY() - LastCursorPosition.y, -Input::GetMouseX() - -LastCursorPosition.x));
	}

	Vector3 MovementDirection = Vector3();
	if (Input::IsKeyDown(26)) {
		MovementDirection += CameraRotation * Vector3(0, 0, 1.F);
	}
	if (Input::IsKeyDown(4)) {
		MovementDirection += CameraRotation * Vector3(1.F, 0, 0);
	}
	if (Input::IsKeyDown(22)) {
		MovementDirection += CameraRotation * Vector3(0, 0, -1.F);
	}
	if (Input::IsKeyDown(7)) {
		MovementDirection += CameraRotation * Vector3(-1.F, 0, 0);
	}

	MovementDirection.y = 0.F;
	MovementDirection.Normalize();
	GetGameObject().LocalTransform.Position += MovementDirection * ViewSpeed * Time::GetDeltaTime<Time::Second>() *
		(!Input::IsKeyDown(225) ? !Input::IsKeyDown(224) ? 1.F : .1F : 4.F);

	GetGameObject().LocalTransform.Rotation = CameraRotation;
}

void CCameraMovement::OnDelete() { }
