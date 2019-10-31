#pragma once

#include "Components/Component.h"

namespace ESource {

	class CCamera : public CComponent {
		IMPLEMENT_COMPONENT(CCamera)
	public:
		virtual void OnRender() override;

		Matrix4x4 GetProjectionMatrix() const;

		float ApertureAngle;

		Vector2 CullingPlanes;

	protected:
		typedef CComponent Supper;
		CCamera(GGameObject & GameObject);

		virtual bool Initialize();

		virtual void OnDelete() override;
	};

}