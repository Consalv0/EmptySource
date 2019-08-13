#pragma once

#include "Events/Event.h"

namespace EmptySource {

	enum class EWindowEventType {
		WindowClose,
		WindowResize,
		WindowGainFocus,
		WindowLostFocus,
		WindowMoved
	};

	class WindowEvent : public Event {
	public:
		virtual EWindowEventType GetEventType() const = 0;

		IMPLEMENT_EVENT_CATEGORY(0u);

	protected:
		WindowEvent() {};
	};

	class WindowCloseEvent : public WindowEvent {
	public:
		WindowCloseEvent() {}

		IMPLEMENT_EVENT_ENUMTYPE(EWindowEventType, WindowClose);
	};

	class WindowResizeEvent : public WindowEvent {
	public:
		WindowResizeEvent(unsigned int Width, unsigned int Height)
			: Width(Width), Height(Height) {}

		inline unsigned int GetWidth() const { return Width; }
		inline unsigned int GetHeight() const { return Height; }

		IMPLEMENT_EVENT_ENUMTYPE(EWindowEventType, WindowResize);

	private:
		unsigned int Width, Height;
	};

	class WindowGainFocusEvent : public WindowEvent {
	public:
		WindowGainFocusEvent() {}

		inline unsigned int GetWidth() const { return Width; }
		inline unsigned int GetHeight() const { return Height; }

		IMPLEMENT_EVENT_ENUMTYPE(EWindowEventType, WindowGainFocus);

	private:
		unsigned int Width, Height;
	};

	class WindowLostFocusEvent : public WindowEvent {
	public:
		WindowLostFocusEvent() {}

		inline unsigned int GetWidth() const { return Width; }
		inline unsigned int GetHeight() const { return Height; }

		IMPLEMENT_EVENT_ENUMTYPE(EWindowEventType, WindowLostFocus);

	private:
		unsigned int Width, Height;
	};

}