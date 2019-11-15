
#include "CoreMinimal.h"
#include "Core/Application.h"
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
	if (Input::IsMouseDown(EMouseButton::Mouse2)) {
		Vector3 EulerAngles = LastCameraRotation.ToEulerAngles();
		CameraRotation = Quaternion::FromEulerAngles(EulerAngles + Vector3(Input::GetMouseY() - LastCursorPosition.Y, -Input::GetMouseX() - -LastCursorPosition.X));
	}

	float AxisY = Input::GetAxis(InputIndex, EJoystickAxis::RightY);
	float AxisX = Input::GetAxis(InputIndex, EJoystickAxis::RightX);
	if (Math::Abs(AxisY) < 0.2F) AxisY = 0.F;
	if (Math::Abs(AxisX) < 0.2F) AxisX = 0.F;
	Vector3 EulerAngles = CameraRotation.ToEulerAngles();
	EulerAngles.Z = 0.F;
	CameraRotation = Quaternion::FromEulerAngles(
		EulerAngles + Vector3(AxisY * ViewSpeed * 0.5F, -AxisX * ViewSpeed, 0.F) * MathConstants::RadToDegree * Time::GetDeltaTime<Time::Second>()
	);

	Vector3 MovementDirection = Vector3();

	if (Input::IsKeyDown(EScancode::W)) {
		MovementDirection += CameraRotation * Vector3(0, 0, 1.F);
	}
	if (Input::IsKeyDown(EScancode::A)) {
		MovementDirection += CameraRotation * Vector3(1.F, 0, 0);
	}
	if (Input::IsKeyDown(EScancode::S)) {
		MovementDirection += CameraRotation * Vector3(0, 0, -1.F);
	}
	if (Input::IsKeyDown(EScancode::D)) {
		MovementDirection += CameraRotation * Vector3(-1.F, 0, 0);
	}

	GetGameObject().LocalTransform.Position.Y += UpVelocity * DeltaTime.GetDeltaTime<Time::Second>();
	UpVelocity -= DeltaTime.GetDeltaTime<Time::Second>() * 15.7F;
	if (GetGameObject().LocalTransform.Position.Y <= DefaultHeight) {
		UpVelocity = 0.F;
		GetGameObject().LocalTransform.Position.Y = DefaultHeight;
	}
	if (InputIndex == 1 && Input::IsButtonDown(InputIndex, EJoystickButton::RightPadDown)) {
		if (DefaultHeight == GetGameObject().LocalTransform.Position.Y) {
			UpVelocity = 5.F;
		}
	}

	AxisY = Input::GetAxis(InputIndex, EJoystickAxis::LeftY);
	AxisX = Input::GetAxis(InputIndex, EJoystickAxis::LeftX);
	if (Math::Abs(AxisY) < 0.2F) AxisY = 0.F;
	if (Math::Abs(AxisX) < 0.2F) AxisX = 0.F;
	MovementDirection += CameraRotation * Vector3(-AxisX, 0, -AxisY);

	float FrameSpeed = ViewSpeed;
	FrameSpeed *= Math::Clamp01(MovementDirection.Magnitude());

	MovementDirection.Y = 0.F;
	MovementDirection.Normalize();
	GetGameObject().LocalTransform.Position += MovementDirection * FrameSpeed * Time::GetDeltaTime<Time::Second>() *
		(Input::IsKeyDown(EScancode::LeftShift) || Input::IsButtonDown(InputIndex, ESource::EJoystickButton::RightPadDown) ? 2.F : 1.F);

	GetGameObject().LocalTransform.Rotation = CameraRotation;
}

void CCameraMovement::OnDelete() { }
