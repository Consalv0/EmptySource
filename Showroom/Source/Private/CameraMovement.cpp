
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "../Public/CameraMovement.h"

CCameraMovement::CCameraMovement(ESource::GGameObject & GameObject)
	: CComponent(L"CameraMovement", GameObject) {
}

void CCameraMovement::OnInputEvent(ESource::InputEvent & InEvent) {
	ESource::EventDispatcher<ESource::InputEvent> Dispatcher(InEvent);
	Dispatcher.Dispatch<ESource::MouseMovedEvent>([this](ESource::MouseMovedEvent & Event) {
		CursorPosition = Event.GetMousePosition();
	});
	Dispatcher.Dispatch<ESource::MouseButtonPressedEvent>([this](ESource::MouseButtonPressedEvent & Event) {
		if (Event.GetMouseButton() == 3 && Event.GetRepeatCount() <= 0) {
			LastCursorPosition = { ESource::Input::GetMouseX(), ESource::Input::GetMouseY() };
			LastCameraRotation = CameraRotation;
		}
	});
}

void CCameraMovement::OnUpdate(const ESource::Timestamp & DeltaTime) {
	const float MinJoystickSensivity = 0.1F;
	if (InputIndex == 1 && ESource::Input::IsMouseDown(ESource::EMouseButton::Mouse2)) {
		Vector3 EulerAngles = LastCameraRotation.ToEulerAngles();
		CameraRotation = Quaternion::FromEulerAngles(EulerAngles + Vector3(ESource::Input::GetMouseY() - LastCursorPosition.Y, -ESource::Input::GetMouseX() - -LastCursorPosition.X));
	}

	float AxisY = ESource::Input::GetAxis(InputIndex, ESource::EJoystickAxis::RightY);
	float AxisX = ESource::Input::GetAxis(InputIndex, ESource::EJoystickAxis::RightX);
	if (Math::Abs(AxisX) < MinJoystickSensivity) AxisX = MinJoystickSensivity;
	if (Math::Abs(AxisY) < MinJoystickSensivity) AxisY = MinJoystickSensivity;
	AxisX = Math::Map(Math::Abs(AxisX), MinJoystickSensivity, 1.F, 0.F, 1.F) * Math::NonZeroSign(AxisX);
	AxisY = Math::Map(Math::Abs(AxisY), MinJoystickSensivity, 1.F, 0.F, 1.F) * Math::NonZeroSign(AxisY);
	ESource::Vector3 EulerAngles = CameraRotation.ToEulerAngles();
	EulerAngles.Z = 0.F;
	CameraRotation = Quaternion::FromEulerAngles(
		EulerAngles + Vector3(AxisY * ViewSpeed, -AxisX * ViewSpeed, 0.F) * MathConstants::RadToDegree * ESource::Time::GetDeltaTime<ESource::Time::Second>()
	);

	Vector3 MovementDirection = Vector3();

	if (InputIndex == 1) {
		if (ESource::Input::IsKeyDown(ESource::EScancode::W)) {
			MovementDirection += CameraRotation * Vector3(0, 0, 1.F);
		}
		if (ESource::Input::IsKeyDown(ESource::EScancode::A)) {
			MovementDirection += CameraRotation * Vector3(1.F, 0, 0);
		}
		if (ESource::Input::IsKeyDown(ESource::EScancode::S)) {
			MovementDirection += CameraRotation * Vector3(0, 0, -1.F);
		}
		if (ESource::Input::IsKeyDown(ESource::EScancode::D)) {
			MovementDirection += CameraRotation * Vector3(-1.F, 0, 0);
		}
	}

	GetGameObject().LocalTransform.Position.Y += UpVelocity * DeltaTime.GetDeltaTime<ESource::Time::Second>();
	UpVelocity -= DeltaTime.GetDeltaTime<ESource::Time::Second>() * 15.7F;
	if (GetGameObject().LocalTransform.Position.Y <= DefaultHeight) {
		UpVelocity = 0.F;
		GetGameObject().LocalTransform.Position.Y = DefaultHeight;
	}
	if (InputIndex == 1) {
		if (ESource::Input::IsKeyDown(ESource::EScancode::Space) || ESource::Input::IsButtonDown(InputIndex, ESource::EJoystickButton::RightPadDown)) {
			if (DefaultHeight == GetGameObject().LocalTransform.Position.Y) {
				UpVelocity = JumpForce;
				JumpForce -= 1.5F;
			}
		}
		JumpForce += DeltaTime.GetDeltaTime<ESource::Time::Second>() * 2.F; 
		JumpForce = Math::Clamp(JumpForce, 1.F, 5.F);
	}

	AxisY = ESource::Input::GetAxis(InputIndex, ESource::EJoystickAxis::LeftY);
	AxisX = ESource::Input::GetAxis(InputIndex, ESource::EJoystickAxis::LeftX);
	if (Math::Abs(AxisX) < MinJoystickSensivity) AxisX = MinJoystickSensivity;
	if (Math::Abs(AxisY) < MinJoystickSensivity) AxisY = MinJoystickSensivity;
	AxisX = Math::Map(Math::Abs(AxisX), MinJoystickSensivity, 1.F, 0.F, 1.F) * Math::NonZeroSign(AxisX);
	AxisY = Math::Map(Math::Abs(AxisY), MinJoystickSensivity, 1.F, 0.F, 1.F) * Math::NonZeroSign(AxisY);
	MovementDirection += CameraRotation * Vector3(-AxisX, 0, -AxisY);

	float FrameSpeed = ViewSpeed;
	FrameSpeed *= Math::Clamp01(MovementDirection.Magnitude());

	MovementDirection.Y = 0.F;
	MovementDirection.Normalize();
	GetGameObject().LocalTransform.Position += MovementDirection * FrameSpeed * ESource::Time::GetDeltaTime<ESource::Time::Second>() *
		(ESource::Input::IsButtonDown(InputIndex, ESource::EJoystickButton::LeftStick) ? 2.F : 1.F);

	GetGameObject().LocalTransform.Rotation = CameraRotation;
}

void CCameraMovement::OnDelete() { }
