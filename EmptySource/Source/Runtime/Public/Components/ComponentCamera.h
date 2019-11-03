#pragma once

#include "Components/Component.h"
#include "Physics/Frustrum.h"

namespace ESource {

	class CCamera : public CComponent {
		IMPLEMENT_COMPONENT(CCamera)
	public:
		Matrix4x4 GetProjectionMatrix() const;

		Frustrum GetFrustrum() const;

		float ApertureAngle;

		Vector2 CullingDistances;

	protected:
		typedef CComponent Supper;
		CCamera(GGameObject & GameObject);

		virtual bool Initialize();
		
		virtual void OnRender() override;

		virtual void OnDelete() override;
	};

}