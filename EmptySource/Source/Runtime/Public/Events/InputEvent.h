#pragma once

#include "Events/Event.h"
#include "Events/KeyCodes.h"
#include "Math/CoreMath.h"

namespace ESource {

	enum class EInputEventType {
		KeyPressed,
		KeyReleased,
		KeyTyped,
		MouseButtonPressed,
		MouseButtonReleased,
		MouseMoved,
		MouseScrolled,
		JoystickConnection,
		JoystickAxis,
		JoystickButtonPressed,
		JoystickButtonReleased,
	};

	enum EInputEventCategory : char {
		None = 0u,
		IEC_Keyboard = 1u << 0,
		IEC_Mouse    = 1u << 1,
		IEC_Joystick = 1u << 2
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
		MouseMovedEvent(float X, float Y, float RelativeX, float RelativeY)
			: MouseX(X), MouseY(Y), MouseRelativeX(RelativeX), MouseRelativeY(RelativeY) {}

		inline float GetX() const { return MouseX; }
		inline float GetY() const { return MouseY; }
		inline float GetXRelative() const { return MouseRelativeX; }
		inline float GetYRelative() const { return MouseRelativeY; }
		inline Point2 GetMousePosition() const { return { MouseX, MouseY }; }
		inline Point2 GetMouseRelativeMotion() const { return { MouseRelativeX, MouseRelativeY }; }

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, MouseMoved)

	private:
		float MouseX, MouseY;
		float MouseRelativeX, MouseRelativeY;
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

	// Joystick Event

	class JoystickEvent : public InputEvent {
	public:
		inline int GetJoystickID() const { return JoystickID; }

		IMPLEMENT_EVENT_CATEGORY(IEC_Joystick)

	protected:
		JoystickEvent(int JoystickID)
			: JoystickID(JoystickID) {
		}

		int JoystickID;
	};

	class JoystickConnectionEvent : public JoystickEvent {
	public:
		JoystickConnectionEvent(int JoystickID, int Connected)
			: JoystickEvent(JoystickID), ConnectionState(Connected) {}

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, JoystickConnection)

		bool IsConnected() const { return ConnectionState; };

	protected:
		int ConnectionState;
	};

	class JoystickButtonPressedEvent : public JoystickEvent {
	public:
		inline EJoystickButton GetButton() const { return ButtonCode; }

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, JoystickButtonPressed)

		JoystickButtonPressedEvent(int JoystickID, EJoystickButton ButtonCode)
			: JoystickEvent(JoystickID), ButtonCode(ButtonCode) {
		}

	protected:
		EJoystickButton ButtonCode;
	};

	class JoystickButtonReleasedEvent : public JoystickEvent {
	public:
		inline EJoystickButton GetButton() const { return ButtonCode; }

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, JoystickButtonReleased)

		JoystickButtonReleasedEvent(int JoystickID, EJoystickButton ButtonCode)
			: JoystickEvent(JoystickID), ButtonCode(ButtonCode) {
		}

	protected:
		EJoystickButton ButtonCode;
	};

	class JoystickAxisEvent : public JoystickEvent {
	public:
		JoystickAxisEvent(int JoystickID, EJoystickAxis Axis, float Value)
			: JoystickEvent(JoystickID), Axis(Axis), Value(Value) {}

		IMPLEMENT_EVENT_ENUMTYPE(EInputEventType, JoystickAxis)

		EJoystickAxis GetAxis() const { return Axis; };

		float GetValue() const { return Value; }

	protected:
		EJoystickAxis Axis;
		float Value;
	};

}