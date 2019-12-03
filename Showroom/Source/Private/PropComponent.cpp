
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

#include "../Public/PropComponent.h"
#include "../Public/ScenePropComponent.h"

#include "Components/ComponentPhysicBody.h"

CProp::CProp(ESource::GGameObject & GameObject)
	: CComponent(L"Gun", GameObject), GameStateComponent(NULL), PhysicBody(NULL), RenderTextureMaterial(L"RenderTextureMaterial") {
	RenderTextureMaterial.DepthFunction = ESource::DF_Always;
	RenderTextureMaterial.CullMode = ESource::CM_None;
	RenderTextureMaterial.SetShaderProgram(ESource::ShaderManager::GetInstance().GetProgram(L"RenderTextureShader"));
}

void CProp::SetPlayerCamera(ESource::CCamera * InPlayerCamera) {
	PlayerCamera = InPlayerCamera;
	StartingTransform = PlayerCamera->GetGameObject().GetParent()->LocalTransform;
}

void CProp::OnUpdate(const ESource::Timestamp & DeltaTime) {
	if (GameStateComponent != NULL && GameStateComponent->GameState == CGameState::EGameState::Started) {
		TArray<ESource::CPhysicBody *> Intersections;
		ESource::Application::GetInstance()->GetPhysicsWorld().AABBIntersection(
			PhysicBody->GetMeshData()->Bounding.Transform(GetGameObject().GetWorldMatrix()), Intersections
		);

		for (auto & Intersection : Intersections) {
			auto SceneProp = Intersection->GetGameObject().GetFirstComponent<CSceneProp>();
			if (SceneProp != NULL) {
				SceneProp->ScaleDown(DeltaTime.GetDeltaTime<ESource::Time::Second>() * 0.1F);
			}
		}
	}
	if (GameStateComponent != NULL && GameStateComponent->GameState == CGameState::EGameState::Starting) {
		PlayerCamera->GetGameObject().GetParent()->LocalTransform.Position =
			Math::Mix(StartingTransform.Position, PlayerCamera->GetGameObject().GetParent()->LocalTransform.Position, GameStateComponent->GetAnimationTime());
	}
}

void CProp::OnPostRender() {
	ESource::Point3 QuadPosition = GetGameObject().GetWorldTransform().Position;
	QuadPosition = (HunterCamera->GetProjectionMatrix() * HunterCamera->GetGameObject().GetWorldTransform().GetGLViewMatrix()).MultiplyPoint(QuadPosition);
	if (QuadPosition.Z < 0.F) {
		QuadPosition.X /= -QuadPosition.Z;
		QuadPosition.Y /= -QuadPosition.Z;
	}
	else {
		QuadPosition.X /= QuadPosition.Z;
		QuadPosition.Y /= QuadPosition.Z;
	}

	{
		ESource::IntBox2D & MainViewport = ESource::Application::GetInstance()->GetWindow().GetViewport();
		MainViewport.MaxY = MainViewport.MaxY / 2;
		ESource::Rendering::SetViewport(MainViewport);
		static float SampleLevel = 0.F;
		static float Gamma = 2.2F;
		int bMonochrome = false;
		int bIsCubemap = false;
		auto & UIArrow = ESource::TextureManager::GetInstance().GetTexture(L"UIArrow");
		RenderTextureMaterial.Use();
		RenderTextureMaterial.SetFloat1Array("_Gamma", &Gamma);
		RenderTextureMaterial.SetInt1Array("_Monochrome", &bMonochrome);
		RenderTextureMaterial.SetFloat4Array("_ColorFilter",
			Vector4(1.F, 1.F, 1.F, (std::cosf(ESource::Time::GetEpochTime<ESource::Time::Second>() * 2.F) + 1.F) / 2.F + .5F).PointerToValue()
		);
		RenderTextureMaterial.SetMatrix4x4Array("_ProjectionMatrix", Matrix4x4().PointerToValue());
		RenderTextureMaterial.SetInt1Array("_IsCubemap", &bIsCubemap);
		UIArrow->GetTexture()->Bind();
		RenderTextureMaterial.SetTexture2D("_MainTexture", UIArrow, 0);
		RenderTextureMaterial.SetTextureCubemap("_MainTextureCube", UIArrow, 1);
		float LODLevel = SampleLevel * (float)UIArrow->GetMipMapCount();
		RenderTextureMaterial.SetFloat1Array("_Lod", &LODLevel);

		ESource::MeshPrimitives::Quad.GetVertexArray()->Bind();
		float ArrowAngle = std::atan2(QuadPosition.X, QuadPosition.Y);
		Vector2 ArrowScale = (Vector2)UIArrow->GetSize() / Vector2((float)MainViewport.GetWidth(), (float)MainViewport.GetHeight());

		if (QuadPosition.Z < 0.F || Math::Abs(QuadPosition.X) > 1.F || Math::Abs(QuadPosition.Y) > 1.F) {
			if (QuadPosition.Z < 0.F) {
				QuadPosition.X = std::sinf(ArrowAngle);
				QuadPosition.Y = std::cosf(ArrowAngle);
				QuadPosition.X = QuadPosition.X * 2.F;
				QuadPosition.Y = QuadPosition.Y * 2.F;
			}

			QuadPosition.X = Math::Clamp(QuadPosition.X, -1.F + ArrowScale.X, 1.F - ArrowScale.X);
			QuadPosition.Y = Math::Clamp(QuadPosition.Y, -1.F + ArrowScale.Y, 1.F - ArrowScale.Y);
			Matrix4x4 QuadScale = Matrix4x4::Scaling(ArrowScale * 0.7F);
			Matrix4x4 QuadRotation = Matrix4x4::Rotation(Vector3(0, 0, -1.F), ArrowAngle);
			QuadPosition.Z = 0.F;
			Matrix4x4 QuadTransformation = Matrix4x4::Translation(QuadPosition) * QuadScale * QuadRotation;
			RenderTextureMaterial.SetMatrix4x4Array("_ModelMatrix", QuadTransformation.PointerToValue());
			ESource::Rendering::DrawIndexed(ESource::MeshPrimitives::Quad.GetVertexArray());
		}
		else if ((HunterCamera->GetGameObject().GetWorldTransform().Position - GetGameObject().GetWorldTransform().Position).MagnitudeSquared() > 100.F) {
			QuadPosition.X = Math::Clamp(QuadPosition.X, -1.F + ArrowScale.X, 1.F - ArrowScale.X);
			QuadPosition.Y = Math::Clamp(QuadPosition.Y, -1.F + ArrowScale.Y, 1.F - ArrowScale.Y);
			Matrix4x4 QuadScale = Matrix4x4::Scaling(ArrowScale * 0.5F);
			QuadPosition.Z = 0.F;
			Matrix4x4 QuadTransformation = Matrix4x4::Translation(QuadPosition) * QuadScale;
			RenderTextureMaterial.SetMatrix4x4Array("_ModelMatrix", QuadTransformation.PointerToValue());
			ESource::Rendering::DrawIndexed(ESource::MeshPrimitives::Quad.GetVertexArray());
		}


		ESource::Rendering::SetViewport(ESource::Application::GetInstance()->GetWindow().GetViewport());
	}
}

void CProp::OnDelete() { }
