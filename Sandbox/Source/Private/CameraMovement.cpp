
#include "CoreMinimal.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "../Public/CameraMovement.h"

using namespace EmptySource;

CCameraMovement::CCameraMovement(EmptySource::GGameObject & GameObject)
	: CComponent(L"CameraMovement", GameObject) {
}

void CCameraMovement::OnInputEvent(EmptySource::InputEvent & InEvent) {
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

void CCameraMovement::OnUpdate(const EmptySource::Timestamp & DeltaTime) {
	if (Input::IsMouseButtonDown(3)) {
		Vector3 EulerAngles = LastCameraRotation.ToEulerAngles();
		CameraRotation = Quaternion::EulerAngles(EulerAngles + Vector3(Input::GetMouseY() - LastCursorPosition.y, -Input::GetMouseX() - -LastCursorPosition.x));
	}

	if (Input::IsKeyDown(26)) {
		Vector3 Forward = CameraRotation * Vector3(0, 0, ViewSpeed);
		GetGameObject().LocalTransform.Position += Forward * Time::GetDeltaTime<Time::Second>() *
			(!Input::IsKeyDown(225) ? !Input::IsKeyDown(224) ? 1.F : .1F : 4.F);
	}
	if (Input::IsKeyDown(4)) {
		Vector3 Right = CameraRotation * Vector3(ViewSpeed, 0, 0);
		GetGameObject().LocalTransform.Position += Right * Time::GetDeltaTime<Time::Second>() *
			(!Input::IsKeyDown(225) ? !Input::IsKeyDown(224) ? 1.F : .1F : 4.F);
	}
	if (Input::IsKeyDown(22)) {
		Vector3 Back = CameraRotation * Vector3(0, 0, -ViewSpeed);
		GetGameObject().LocalTransform.Position += Back * Time::GetDeltaTime<Time::Second>() *
			(!Input::IsKeyDown(225) ? !Input::IsKeyDown(224) ? 1.F : .1F : 4.F);
	}
	if (Input::IsKeyDown(7)) {
		Vector3 Left = CameraRotation * Vector3(-ViewSpeed, 0, 0);
		GetGameObject().LocalTransform.Position += Left * Time::GetDeltaTime<Time::Second>() *
			(!Input::IsKeyDown(225) ? !Input::IsKeyDown(224) ? 1.F : .1F : 4.F);
	}

	GetGameObject().LocalTransform.Rotation = CameraRotation;
}

void CCameraMovement::OnDelete() { }
