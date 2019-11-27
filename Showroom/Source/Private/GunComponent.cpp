
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Core/GameObject.h"
#include "Core/Transform.h"
#include "Core/Input.h"

#include "Resources/ModelResource.h"
#include "Rendering/MeshPrimitives.h"
#include "Resources/TextureManager.h"
#include "Resources/ShaderManager.h"
#include "Physics/PhysicsWorld.h"
#include "Physics/Ray.h"

#include "../Public/GunComponent.h"
#include "../Public/PropComponent.h"
#include "Components/ComponentPhysicBody.h"

CGun::CGun(ESource::GGameObject & GameObject)
	: CComponent(L"Gun", GameObject), RenderTextureMaterial(L"RenderTextureMaterial") {
	RenderTextureMaterial.DepthFunction = ESource::DF_Always;
	RenderTextureMaterial.CullMode = ESource::CM_None;
	RenderTextureMaterial.SetShaderProgram(ESource::ShaderManager::GetInstance().GetProgram(L"RenderTextureShader"));
}

void CGun::SetGunObjects(ESource::GGameObject * Gun, ESource::CAnimable * Animable, ESource::CCamera * Camera) {
	GunObject = Gun;
	ES_CORE_ASSERT(GunObject != NULL, "GunObject is NULL");
	GunAnimable = Animable;
	ES_CORE_ASSERT(GunAnimable != NULL, "GunAnimable is NULL");
	PlayerCamera = Camera;
	ES_CORE_ASSERT(PlayerCamera != NULL, "PlayerCamera is NULL");
	GunAnimable->AddEventOnEndAnimation("GunComponent", [this]() { bReloading = false; LOG_CORE_DEBUG("Animation Ended"); });
}

void CGun::OnUpdate(const ESource::Timestamp & DeltaTime) {
	ESource::TArray<ESource::RayHit> Hits;
	ESource::Vector3 CameraRayDirection = {
		(ESource::Application::GetInstance()->GetWindow().GetWidth()) / ESource::Application::GetInstance()->GetWindow().GetWidth() - 1.F,
		1.F - (ESource::Application::GetInstance()->GetWindow().GetHeight()) / ESource::Application::GetInstance()->GetWindow().GetHeight(),
		-1.F,
	};
	CameraRayDirection = PlayerCamera->GetProjectionMatrix().Inversed() * CameraRayDirection;
	CameraRayDirection.Z = -1.F;
	CameraRayDirection = PlayerCamera->GetGameObject().LocalTransform.GetGLViewMatrix().Inversed() * CameraRayDirection;
	CameraRayDirection.Normalize();
	ESource::Ray CameraRay(PlayerCamera->GetGameObject().GetWorldTransform().Position, CameraRayDirection);
	ESource::Application::GetInstance()->GetPhysicsWorld().RayCast(CameraRay, Hits);
	if (!Hits.empty()) {
		GetGameObject().LocalTransform.Rotation =
			(GetGameObject().IsRoot() ? ESource::Quaternion() : GetGameObject().GetParent()->GetWorldTransform().Rotation.Inversed()) *
			ESource::Quaternion::FromLookRotation(CameraRay.PointAt(Hits[0].Stamp) - GetGameObject().GetWorldTransform().Position, { 0.F, 1.F, 0.F });
		if (ESource::Input::IsMousePressed(ESource::EMouseButton::Mouse0) || ESource::Input::GetAxis(-1, ESource::EJoystickAxis::TriggerRight) > 0.5F) {
			if (!bReloading) {
				LOG_CORE_DEBUG(L"Hit with: {}", Hits[0].PhysicBody->GetGameObject().GetName().GetInstanceName().c_str());
				if (Hits[0].PhysicBody->GetGameObject().GetFirstComponent<CProp>() != NULL) {
					LOG_CORE_CRITICAL(L"Hunted!");
					ESource::Input::SendHapticImpulse(0, 0, 0.5F, 200);
					ESource::Input::SendHapticImpulse(1, 0, 0.5F, 200);
					(++ESource::Application::GetInstance()->GetAudioDevice().begin())->second->Pos = 0;
					(++ESource::Application::GetInstance()->GetAudioDevice().begin())->second->bPause = false;
				}
			}
		}
	}

	if (ESource::Input::IsMouseDown(ESource::EMouseButton::Mouse0) || ESource::Input::GetAxis(-1, ESource::EJoystickAxis::TriggerRight) > 0.5F) {
		if (!bReloading) {
			bReloading = true;
			GunAnimable->bPlaying = true;
			ESource::Application::GetInstance()->GetAudioDevice().begin()->second->Pos = 0;
			ESource::Application::GetInstance()->GetAudioDevice().begin()->second->bPause = false;
		}
	}
}

void CGun::OnPostRender() {
	{ // Render crosshead
		auto & MainViewport = ESource::Application::GetInstance()->GetWindow().GetViewport();
		MainViewport.MaxY = MainViewport.MaxY / 2;
		ESource::Rendering::SetViewport(MainViewport);
		static float SampleLevel = 0.F;
		static float Gamma = 2.2F;
		static bool ColorFilter[4] = { true, true, true, true };
		int bMonochrome = false;
		int bIsCubemap = false;
		auto & CrossHead = ESource::TextureManager::GetInstance().GetTexture(L"CrossHead");
		RenderTextureMaterial.Use();
		RenderTextureMaterial.SetFloat1Array("_Gamma", &Gamma);
		RenderTextureMaterial.SetInt1Array("_Monochrome", &bMonochrome);
		RenderTextureMaterial.SetFloat4Array("_ColorFilter",
			Vector4(ColorFilter[0] ? 1.F : 0.F, ColorFilter[1] ? 1.F : 0.F, ColorFilter[2] ? 1.F : 0.F, ColorFilter[3] ? 1.F : 0.F)
			.PointerToValue()
		);
		RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
		RenderTextureMaterial.SetInt1Array("_IsCubemap", &bIsCubemap);
		CrossHead->GetTexture()->Bind();
		RenderTextureMaterial.SetTexture2D("_MainTexture", CrossHead, 0);
		RenderTextureMaterial.SetTextureCubemap("_MainTextureCube", CrossHead, 1);
		float LODLevel = SampleLevel * (float)CrossHead->GetMipMapCount();
		RenderTextureMaterial.SetFloat1Array("_Lod", &LODLevel);

		ESource::MeshPrimitives::Quad.GetVertexArray()->Bind();
		Matrix4x4 QuadPosition = Matrix4x4::Scaling((Vector2)CrossHead->GetSize() / Vector2((float)MainViewport.GetWidth(), (float)MainViewport.GetHeight()));
		RenderTextureMaterial.SetMatrix4x4Array("_ModelMatrix", QuadPosition.PointerToValue());

		ESource::Rendering::DrawIndexed(ESource::MeshPrimitives::Quad.GetVertexArray());
		ESource::Rendering::SetViewport(ESource::Application::GetInstance()->GetWindow().GetViewport());
	}
}

void CGun::OnDelete() { }