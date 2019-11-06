#pragma once

#include "Components/Component.h"

namespace ESource {

	class CLight : public CComponent {
		IMPLEMENT_COMPONENT(CLight)
	public:
		virtual void OnRender() override;

		void SetShadowMapSize(int Size);

		Vector3 Color;
		float Intensity;
		bool bCastShadow;
		float ShadowMapBias;
		float ApertureAngle;
		Vector2 CullingPlanes;
		uint8_t RenderingMask;

	protected:
		typedef CComponent Supper;
		CLight(GGameObject & GameObject);

		virtual bool Initialize();

		virtual void OnDelete() override;

		int ShadowMapSize;

		RTexturePtr ShadowMap;

	};

}