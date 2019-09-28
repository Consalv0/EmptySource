
#include "CoreMinimal.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Resources/MeshManager.h"
#include "Resources/TextureManager.h"
#include "Core/GameObject.h"
#include "Core/Application.h"

#include "Components/ComponentLight.h"

namespace ESource {

	CLight::CLight(GGameObject & GameObject)
		: CComponent(L"CLight", GameObject), Color(0.9F, 1.0F, 1.0F), Intensity(20.F),
		bCastShadow(false), ShadowMapBias(0.5F), ShadowMap(NULL), ShadowMapSize(512),
		ApertureAngle(60.F), CullingPlanes(0.01F, 1000.F) {
	}

	void CLight::SetShadowMapSize(int Size) {
		if (ShadowMapSize == Size) return;
		ShadowMapSize = Size;
		if (ShadowMap) {
			ShadowMap->Unload();
			ShadowMap->SetSize({ ShadowMapSize, ShadowMapSize, 1 });
			ShadowMap->Load();
		}
	}

	void CLight::OnRender() {
		RenderPipeline & AppRenderPipeline = Application::GetInstance()->GetRenderPipeline();
		AppRenderPipeline.SubmitDirectionalLight(0, GetGameObject().GetWorldTransform(), Color, Intensity,
			Matrix4x4::Perspective(ApertureAngle * MathConstants::DegreeToRad, 1.F, CullingPlanes.x, CullingPlanes.y)
		);
		if (bCastShadow) {
			if (ShadowMap == NULL)
				ShadowMap = TextureManager::GetInstance().CreateTexture2D(
					GetName().GetInstanceName() + L"_ShadowMap", L"", PF_ShadowDepth, FM_MinMagNearest, SAM_Border, ShadowMapSize
				);
			AppRenderPipeline.SubmitDirectionalShadowMap(0, ShadowMap, ShadowMapBias);
		}
	}
	
	bool CLight::Initialize() {
		return true;
	}
	
	void CLight::OnDelete() {
		TextureManager::GetInstance().FreeTexture(GetName().GetInstanceName() + L"_ShadowMap");
	}

}