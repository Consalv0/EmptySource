#pragma once

#include "CoreTypes.h"
#include "Events/WindowEvent.h"
#include "Events/InputEvent.h"
#include "Core/CoreTime.h"
#include "Core/IIdentifier.h"

namespace EmptySource {

	class Layer : public IIdentifier {
	public:
		Layer(const WString& Name, unsigned int Level);

		virtual ~Layer() = default;

		virtual void OnAttach() {}

		virtual void OnAwake() {}

		virtual void OnDetach() {}

		virtual void OnRender() {}
		
		virtual void OnUpdate(Timestamp Stamp) {}
		
		virtual void OnImGuiRender() {}
		
		virtual void OnWindowEvent(WindowEvent& WinEvent) {}
		
		virtual void OnInputEvent(InputEvent& InEvent) {}

		inline WString GetName() { return Name; }

		inline unsigned int GetLevel() const { return Level; }

		inline bool operator < (const Layer& Other) const {
			return (GetLevel() < Other.GetLevel());
		}
		inline bool operator <= (const Layer& Other) const {
			return (GetLevel() <= Other.GetLevel());
		}
		inline bool operator > (const Layer& Other) const {
			return (GetLevel() > Other.GetLevel());
		}
		inline bool operator >= (const Layer& Other) const {
			return (GetLevel() >= Other.GetLevel());
		}

	protected:
		virtual inline int GetLayerPriority() { return 0; };

	private:
		WString Name;
		unsigned int Level;

	};

}