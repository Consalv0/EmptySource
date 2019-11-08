
#include "CoreMinimal.h"
#include "Rendering/RenderStage.h"
#include "Rendering/RenderPipeline.h"
#include "Resources/TextureManager.h"
#include "Core/GameObject.h"
#include "Core/Application.h"

#include "Components/ComponentLight.h"

namespace ESource {

	CLight::CLight(GGameObject & GameObject)
		: CComponent(L"CLight", GameObject), Color(0.9F, 1.0F, 1.0F), Intensity(20.F),
		bCastShadow(false), ShadowMapBias(0.001F), ShadowMap(NULL), ShadowMapSize(1024),
		ApertureAngle(120.F), CullingPlanes(0.3F, 1000.F), RenderingMask(1) {
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
		if (bCastShadow) {
			if (ShadowMap == NULL)
				ShadowMap = TextureManager::GetInstance().CreateTexture2D(
					GetName().GetInstanceName() + L"_ShadowMap", L"", PF_ShadowDepth, FM_MinMagLinear, SAM_Clamp, ShadowMapSize
				);
		}
		else if (ShadowMap != NULL) {
			ShadowMap->Unload();
		}

		RenderPipeline & AppRenderPipeline = Application::GetInstance()->GetRenderPipeline();
		AppRenderPipeline.SubmitSpotLight(GetGameObject().GetWorldTransform(), Color, GetGameObject().GetWorldTransform().Forward(), Intensity,
			Matrix4x4::Perspective(ApertureAngle * MathConstants::DegreeToRad, 1.F, CullingPlanes.X, CullingPlanes.Y),
			bCastShadow ? ShadowMap : NULL, ShadowMapBias, RenderingMask
		);
	}
	
	bool CLight::Initialize() {
		return true;
	}
	
	void CLight::OnDelete() {
		TextureManager::GetInstance().FreeTexture(GetName().GetInstanceName() + L"_ShadowMap");
	}

}