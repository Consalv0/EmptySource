#pragma once

#include "CoreTypes.h"
#include "Events/WindowEvent.h"
#include "Engine/CoreTime.h"

namespace EmptySource {

	class Layer {
	public:
		Layer(const WString& Name);

		virtual ~Layer() = default;

		virtual void OnAttach() {};
		virtual void OnDetach() {}
		virtual void OnUpdate(Timestamp Stamp) {}
		virtual void OnImGuiRender() {}
		virtual void OnWindowEvent(WindowEvent& event) {}

	};

}