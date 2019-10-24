#pragma once

#include "Components/Component.h"
#include "Rendering/Animation.h"

namespace ESource {

	class CAnimable : public CComponent {
		IMPLEMENT_COMPONENT(CAnimable)
	public:
		virtual void OnUpdate(const Timestamp& Stamp) override;

		const AnimationTrack * Track;

		// Current animation time
		double CurrentAnimationTime;

		double AnimationSpeed;

	protected:
		typedef CComponent Supper;
		CAnimable(GGameObject & GameObject);

		virtual bool Initialize();

		virtual void OnDelete() override;

		void UpdateHierarchy();

		GGameObject * FindInHiererchy(GGameObject * GO, const IName & Name);
	};

}