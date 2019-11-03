
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
	if (Input::IsMouseDown(MouseButton::Mouse2)) {
		Vector3 EulerAngles = LastCameraRotation.ToEulerAngles();
		CameraRotation = Quaternion::FromEulerAngles(EulerAngles + Vector3(Input::GetMouseY() - LastCursorPosition.Y, -Input::GetMouseX() - -LastCursorPosition.X));
	}

	Vector3 MovementDirection = Vector3();

	if (Input::IsKeyDown(Scancode::W)) {
		MovementDirection += CameraRotation * Vector3(0, 0, 1.F);
	}
	if (Input::IsKeyDown(Scancode::A)) {
		MovementDirection += CameraRotation * Vector3(1.F, 0, 0);
	}
	if (Input::IsKeyDown(Scancode::S)) {
		MovementDirection += CameraRotation * Vector3(0, 0, -1.F);
	}
	if (Input::IsKeyDown(Scancode::D)) {
		MovementDirection += CameraRotation * Vector3(-1.F, 0, 0);
	}

	MovementDirection.Y = 0.F;
	MovementDirection.Normalize();
	GetGameObject().LocalTransform.Position += MovementDirection * ViewSpeed * Time::GetDeltaTime<Time::Second>() *
		(!Input::IsKeyDown(Scancode::LeftShift) ? !Input::IsKeyDown(Scancode::LeftCtrl) ? 1.F : .1F : 4.F);

	GetGameObject().LocalTransform.Rotation = CameraRotation;

	if (Input::IsMousePressed(MouseButton::Mouse0)) {
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
