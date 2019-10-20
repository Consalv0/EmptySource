#pragma once

#include "CoreTypes.h"
#include "Events/WindowEvent.h"
#include "Events/InputEvent.h"
#include "Core/CoreTime.h"

namespace ESource {

	class Layer {
	public:
		Layer(const IName& Name, uint32_t Level);

		virtual ~Layer() = default;

		virtual void OnAttach() {}

		virtual void OnAwake() {}

		virtual void OnDetach() {}

		virtual void OnRender() {}
		
		virtual void OnUpdate(Timestamp Stamp) {}
		
		virtual void OnImGuiRender() {}
		
		virtual void OnWindowEvent(WindowEvent& WinEvent) {}
		
		virtual void OnInputEvent(InputEvent& InEvent) {}

		inline const IName & GetName() { return Name; }

		inline uint32_t GetLayerPriority() const { return Level; }

		inline bool operator < (const Layer& Other) const {
			return (GetLayerPriority() < Other.GetLayerPriority());
		}
		inline bool operator <= (const Layer& Other) const {
			return (GetLayerPriority() <= Other.GetLayerPriority());
		}
		inline bool operator > (const Layer& Other) const {
			return (GetLayerPriority() > Other.GetLayerPriority());
		}
		inline bool operator >= (const Layer& Other) const {
			return (GetLayerPriority() >= Other.GetLayerPriority());
		}

	protected:
		const IName Name;
		uint32_t Level;

	};

}