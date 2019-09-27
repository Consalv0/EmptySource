#pragma once

#include "Core/Object.h"

#define IMPLEMENT_COMPONENT(Name) public: \
inline virtual ESource::WString GetObjectName() override { return L#Name;} \
static ESource::WString GetStaticObjectName() { return L#Name;} \
protected: \
friend class ESource::GGameObject; friend class ESource::SpaceLayer; 

namespace ESource {

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

		GGameObject & Holder;

	private:

		virtual void OnDelete() override;

		virtual void OnDetach() override {};

	};

}