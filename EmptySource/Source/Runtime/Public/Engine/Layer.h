#pragma once

#include "CoreTypes.h"
#include "Events/WindowEvent.h"
#include "Events/InputEvent.h"
#include "Engine/CoreTime.h"
#include "Engine/IIdentifier.h"

namespace EmptySource {

	class Layer : public IIdentifier {
	public:
		Layer(const WString& Name);

		virtual ~Layer() = default;

		virtual void OnAttach() {}

		virtual void OnAwake() {}

		virtual void OnDetach() {}

		virtual void OnRender() {}
		
		virtual void OnUpdate(Timestamp Stamp) {}

		virtual void OnTerminate() {}
		
		virtual void OnImGuiRender() {}
		
		virtual void OnWindowEvent(WindowEvent& WinEvent) {}
		
		virtual void OnInputEvent(InputEvent& InEvent) {}

		inline WString GetName() { return Name; }

	protected:
		virtual inline int GetLayerPriority() { return 0; };

	private:
		WString Name;

	};

}