
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
	if (Input::IsMouseDown(MouseButton::Mouse2)) {
		Vector3 EulerAngles = LastCameraRotation.ToEulerAngles();
		CameraRotation = Quaternion::EulerAngles(EulerAngles + Vector3(Input::GetMouseY() - LastCursorPosition.y, -Input::GetMouseX() - -LastCursorPosition.x));
	}

	if (Input::IsKeyDown(Scancode::W)) {
		Vector3 Forward = CameraRotation * Vector3(0, 0, ViewSpeed);
		GetGameObject().LocalTransform.Position += Forward * Time::GetDeltaTime<Time::Second>() *
			(!Input::IsKeyDown(Scancode::LSHIFT) ? !Input::IsKeyDown(Scancode::LCTRL) ? 1.F : .1F : 4.F);
	}
	if (Input::IsKeyDown(Scancode::A)) {
		Vector3 Right = CameraRotation * Vector3(ViewSpeed, 0, 0);
		GetGameObject().LocalTransform.Position += Right * Time::GetDeltaTime<Time::Second>() *
			(!Input::IsKeyDown(Scancode::LSHIFT) ? !Input::IsKeyDown(Scancode::LCTRL) ? 1.F : .1F : 4.F);
	}
	if (Input::IsKeyDown(Scancode::S)) {
		Vector3 Back = CameraRotation * Vector3(0, 0, -ViewSpeed);
		GetGameObject().LocalTransform.Position += Back * Time::GetDeltaTime<Time::Second>() *
			(!Input::IsKeyDown(Scancode::LSHIFT) ? !Input::IsKeyDown(Scancode::LCTRL) ? 1.F : .1F : 4.F);
	}
	if (Input::IsKeyDown(Scancode::D)) {
		Vector3 Left = CameraRotation * Vector3(-ViewSpeed, 0, 0);
		GetGameObject().LocalTransform.Position += Left * Time::GetDeltaTime<Time::Second>() *
			(!Input::IsKeyDown(Scancode::LSHIFT) ? !Input::IsKeyDown(Scancode::LCTRL) ? 1.F : .1F : 4.F);
	}

	GetGameObject().LocalTransform.Rotation = CameraRotation;
}

void CCameraMovement::OnDelete() { }
