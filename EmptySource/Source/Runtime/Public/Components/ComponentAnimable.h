#pragma once

#include "Components/Component.h"
#include "Rendering/Animation.h"

namespace ESource {

	class CAnimable : public CComponent {
		IMPLEMENT_COMPONENT(CAnimable)
	public:
		typedef std::function<void()> CallbackFunctionPointer; // Typedef for a function pointer

		virtual void OnUpdate(const Timestamp& Stamp) override;

		void AddEventOnEndAnimation(const NString& Name, const CallbackFunctionPointer & Function);

		const AnimationTrack * Track;

		bool bLoop;

		bool bPlaying;

		// Current animation time
		double CurrentAnimationTime;

		double AnimationSpeed;

	protected:
		typedef CComponent Supper;
		CAnimable(GGameObject & GameObject);

		TDictionary<NString, CallbackFunctionPointer> EventsCallback;

		virtual bool Initialize();

		virtual void OnDelete() override;

		void CallEventsOnEndAnimation();

		void UpdateHierarchy();

		GGameObject * FindInHiererchy(GGameObject * GO, const IName & Name);
	};

}