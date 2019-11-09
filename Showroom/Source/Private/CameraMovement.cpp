
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "Resources/ModelResource.h"

#include "Components/ComponentCamera.h"
#include "Components/ComponentPhysicBody.h"

#include "Physics/PhysicsWorld.h"
#include "Physics/Ray.h"

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

	float AxisY = Input::GetAxis(-1, EJoystickAxis::RightY);
	float AxisX = Input::GetAxis(-1, EJoystickAxis::RightX);
	if (Math::Abs(AxisY) < 0.2F) AxisY = 0.F;
	if (Math::Abs(AxisX) < 0.2F) AxisX = 0.F;
	CameraRotation = Quaternion::FromEulerAngles(
		CameraRotation.ToEulerAngles() + Vector3(AxisY * ViewSpeed * 0.5F, -AxisX * ViewSpeed, 0.F) * MathConstants::RadToDegree * Time::GetDeltaTime<Time::Second>()
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

	AxisY = Input::GetAxis(-1, EJoystickAxis::LeftY);
	AxisX = Input::GetAxis(-1, EJoystickAxis::LeftX);
	if (Math::Abs(AxisY) < 0.2F) AxisY = 0.F;
	if (Math::Abs(AxisX) < 0.2F) AxisX = 0.F;
	MovementDirection += CameraRotation * Vector3(-AxisX, 0, -AxisY);

	float FrameSpeed = ViewSpeed;
	FrameSpeed *= Math::Clamp01(MovementDirection.Magnitude());

	MovementDirection.Y = 0.F;
	MovementDirection.Normalize();
	GetGameObject().LocalTransform.Position += MovementDirection * FrameSpeed * Time::GetDeltaTime<Time::Second>() *
		(!Input::IsKeyDown(EScancode::LeftShift) ? !Input::IsKeyDown(EScancode::LeftCtrl) ? 1.F : .1F : 4.F);

	GetGameObject().LocalTransform.Rotation = CameraRotation;

	if (Input::IsMousePressed(EMouseButton::Mouse0) || Input::IsButtonPressed(-1, EJoystickButton::RightPadDown)) {
		TArray<RayHit> Hits;
		Vector3 CameraRayDirection = {
			(2.F * Input::GetMouseX()) / Application::GetInstance()->GetWindow().GetWidth() - 1.F,
			1.F - (2.F * Input::GetMouseY()) / Application::GetInstance()->GetWindow().GetHeight(),
			-1.F,
		};
		CameraRayDirection = GetGameObject().GetFirstComponent<CCamera>()->GetProjectionMatrix().Inversed() * CameraRayDirection;
		CameraRayDirection.Z = -1.F;
		CameraRayDirection = GetGameObject().LocalTransform.GetGLViewMatrix().Inversed() * CameraRayDirection;
		CameraRayDirection.Normalize();
		Ray CameraRay(GetGameObject().GetWorldTransform().Position, CameraRayDirection);
		Application::GetInstance()->GetPhysicsWorld().RayCast(CameraRay, Hits);
		for (auto & Hit : Hits) {
			LOG_CORE_DEBUG(L"Hit with: {}", Hit.PhysicBody->GetGameObject().GetName().GetInstanceName().c_str());
		}
	}
}

void CCameraMovement::OnDelete() { }
