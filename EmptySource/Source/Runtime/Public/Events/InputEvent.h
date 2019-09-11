#pragma once

#include "Events/Event.h"
#include "Math/CoreMath.h"

namespace EmptySource {

	enum class EInputEventType {
		KeyPressed,
		KeyReleased,
		KeyTyped,
		MouseButtonPressed,
		MouseButtonReleased,
		MouseMoved,
		MouseScrolled
	};

	enum EInputEventCategory : unsigned int {
		None = 0u,
		IEC_Keyboard = 1u << 0,
		IEC_Mouse    = 1u << 1,
		IEC_Gamepad  = 1u << 2
	};

	class InputEvent : public Event {
	public:
		virtual EInputEventType GetEventType() const = 0;

		inline bool IsInCategory(EInputEventCategory Category) {
			return GetCategoryFlags() & Category;
		}
	};

	// Key Events //

	class KeyEvent : public InputEvent {
	public:
		inline int GetKeyCode() const { return EventKeyCode; }

		inline bool GetKeyShiftModifier() const { return ModKeyShift; };
		inline bool GetKeyCtrlModifier() const { return ModKeyCtrl; };
		inline bool GetKeyAltModifier() const { return ModKeyAlt; };
		inline bool GetKeySuperModifier() const { return ModKeySuper; };

		IMPLEMENT_EVENT_CATEGORY(IEC_Keyboard)

	protected:
		KeyEvent(int Code, bool Shift, bool Ctrl, bool Alt, bool Super)
			: EventKeyCode(Code), ModKeyShift(Shift), ModKeyCtrl(Ctrl), ModKeyAlt(Alt), ModKeySuper(Super) {
		}

		int EventKeyCode;
		bool ModKeyShift; bool ModKeyCtrl; bool ModKeyAlt; bool ModKeySuper;
	};

	class KeyPressedEvent : public KeyEvent {
	public:
		KeyPressedEvent(int Code, bool Shift, bool Ctrl, bool Alt, bool Super, bool Repeated)
			: KeyEvent(Code, Shift, Ctrl, Alt, Super), bRepeat(Repeated) {}

		//* The key event was fired by maintaining the key pressed?
		inline bool IsRepeated() const { return bRepeat; }

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, KeyPressed)

	private:
		bool bRepeat;
	};

	class KeyReleasedEvent : public KeyEvent {
	public:
		KeyReleasedEvent(int Code, bool Shift, bool Ctrl, bool Alt, bool Super)
			: KeyEvent(Code, Shift, Ctrl, Alt, Super) {}

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, KeyReleased)
	};

	class KeyTypedEvent : public KeyEvent {
	public:
		KeyTypedEvent(const NChar Text[32])
			: KeyEvent(0, false, false, false, false) {
			for (size_t i = 0; i < 32; ++i) EventText[i] = Text[i];
		}

		inline NString GetText() const { return EventText; }

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, KeyTyped)

	private:
		NChar EventText[32];
	};

	// Mouse Events //

	class MouseEvent : public InputEvent {
	public:
		IMPLEMENT_EVENT_CATEGORY(IEC_Mouse)

	protected:
		MouseEvent() {}
	};


	class MouseMovedEvent : public MouseEvent {
	public:
		MouseMovedEvent(float x, float y, float xRelative, float yRelative)
			: MouseX(x), MouseY(y), MouseXRelative(xRelative), MouseYRelative(yRelative) {}

		inline float GetX() const { return MouseX; }
		inline float GetY() const { return MouseY; }
		inline float GetXRelative() const { return MouseXRelative; }
		inline float GetYRelative() const { return MouseYRelative; }
		inline Point2 GetMousePosition() const { return { MouseX, MouseY }; }
		inline Point2 GetMouseRelativeMotion() const { return { MouseXRelative, MouseYRelative }; }

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, MouseMoved)

	private:
		float MouseX, MouseY;
		float MouseXRelative, MouseYRelative;
	};

	class MouseScrolledEvent : public MouseEvent {
	public:
		MouseScrolledEvent(float OffsetX, float OffsetY, bool Flipped)
			: OffsetX(OffsetX), OffsetY(OffsetY), bFlipped(Flipped) {}

		inline float GetOffsetX() const { return OffsetX; }
		inline float GetOffsetY() const { return OffsetY; }
		inline bool IsFlipped() const { return bFlipped; }

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, MouseScrolled)
	
	private:
		float OffsetX, OffsetY;
		bool bFlipped;
	};

	class MouseButtonEvent : public MouseEvent {
	public:
		inline int GetMouseButton() const { return EventButton; }

		inline int IsDoubleClick() const { return bDoubleClick; }

		inline int GetRepeatCount() const { return RepeatCount; }

	protected:
		MouseButtonEvent(int Button, bool DoubleClick, int RepeatCount)
			: EventButton(Button), bDoubleClick(DoubleClick), RepeatCount(RepeatCount) {}

		int EventButton;
		bool bDoubleClick;
		int RepeatCount;
	};

	class MouseButtonPressedEvent : public MouseButtonEvent {
	public:
		MouseButtonPressedEvent(int Button, bool DoubleClick, int RepeatCount)
			: MouseButtonEvent(Button, DoubleClick, RepeatCount) {}

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, MouseButtonPressed)
	};

	class MouseButtonReleasedEvent : public MouseButtonEvent {
	public:
		MouseButtonReleasedEvent(int Button)
			: MouseButtonEvent(Button, false, 0) {}
		
		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, MouseButtonReleased)
	};

}