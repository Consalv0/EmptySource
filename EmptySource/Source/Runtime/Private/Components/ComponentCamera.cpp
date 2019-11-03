
#include "CoreMinimal.h"
#include "Rendering/Texture.h"
#include "Rendering/Material.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Core/GameObject.h"
#include "Core/CoreTime.h"
#include "Core/Application.h"

#include "Components/ComponentCamera.h"

namespace ESource {

	Matrix4x4 CCamera::GetProjectionMatrix() const {
		return Matrix4x4::Perspective(
			ApertureAngle * MathConstants::DegreeToRad,
			Application::GetInstance()->GetWindow().GetAspectRatio(),
			CullingDistances.X, CullingDistances.Y
		);
	}
	
	Frustrum CCamera::GetFrustrum() const {
		Matrix4x4 ComboMatrix = GetProjectionMatrix() * GetGameObject().GetWorldTransform().GetGLViewMatrix().Transposed();
		return Frustrum::FromProjectionViewMatrix(ComboMatrix);
	}

	CCamera::CCamera(GGameObject & GameObject)
		: CComponent(L"Camera", GameObject), ApertureAngle(60.F), CullingDistances(0.03F, 1000.F) {
	}

	void CCamera::OnRender() {
		Application::GetInstance()->GetRenderPipeline().SetEyeTransform(GetGameObject().GetWorldTransform());
		Application::GetInstance()->GetRenderPipeline().SetProjectionMatrix(GetProjectionMatrix());
	}

	bool CCamera::Initialize() {
		return true;
	}

	void CCamera::OnDelete() {
	}

}