#pragma once

#include "Core/Object.h"

#define IMPLEMENT_COMPONENT(Name) public: \
inline virtual WString GetObjectName() override { return L#Name;} \
static WString GetStaticObjectName() { return L#Name;} \
protected: \
friend class GGameObject; friend class SpaceLayer; 

namespace EmptySource {

	class GGameObject;

	class CComponent : public OObject {
		IMPLEMENT_COMPONENT(CComponent)
	public:
		GGameObject & GetGameObject() const;

	protected:
		typedef OObject Supper;
		
		CComponent(GGameObject & GameObject);

		CComponent(const IName & Name, GGameObject & GameObject);

		virtual void OnRender() {};

		virtual void OnUpdate(const Timestamp& Stamp) {};

		virtual void OnImGuiRender() {};

		virtual void OnWindowEvent(WindowEvent& WinEvent) {};

		virtual void OnInputEvent(InputEvent& InEvent) {};

		virtual void OnDelete() override;

		GGameObject & Holder;

	};

}