
#include "CoreMinimal.h"
#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Resources/MeshManager.h"
#include "Core/GameObject.h"
#include "Core/CoreTime.h"
#include "Core/Application.h"

#include "Components/ComponentCamera.h"

namespace ESource {

	CCamera::CCamera(GGameObject & GameObject) 
		: CComponent(L"Camera", GameObject), ApertureAngle(60.F), CullingPlanes(0.03F, 1000.F) {
	}

	void CCamera::OnRender() {
		Application::GetInstance()->GetRenderPipeline().SetEyeTransform(GetGameObject().GetWorldTransform());
		Application::GetInstance()->GetRenderPipeline().SetProjectionMatrix(
			Matrix4x4::Perspective(
			ApertureAngle * MathConstants::DegreeToRad,
			Application::GetInstance()->GetWindow().GetAspectRatio(),
			CullingPlanes.x, CullingPlanes.y
		));
	}

	bool CCamera::Initialize() {
		LOG_CORE_DEBUG(L"Camera '{0}'[{1:d}] Initalized", Name.GetDisplayName(), Name.GetInstanceID());
		return true;
	}

	void CCamera::OnDelete() {
		LOG_CORE_DEBUG(L"Camera '{0}'[{1:d}] Destroyed", Name.GetDisplayName(), Name.GetInstanceID());
	}

}