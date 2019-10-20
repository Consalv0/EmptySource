#pragma once

#include "Events/Event.h"

namespace ESource {

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
		WindowResizeEvent(uint32_t Width, uint32_t Height)
			: Width(Width), Height(Height) {}

		inline uint32_t GetWidth() const { return Width; }
		inline uint32_t GetHeight() const { return Height; }

		IMPLEMENT_EVENT_ENUMTYPE(EWindowEventType, WindowResize);

	private:
		uint32_t Width, Height;
	};

	class WindowGainFocusEvent : public WindowEvent {
	public:
		WindowGainFocusEvent() {}

		inline uint32_t GetWidth() const { return Width; }
		inline uint32_t GetHeight() const { return Height; }

		IMPLEMENT_EVENT_ENUMTYPE(EWindowEventType, WindowGainFocus);

	private:
		uint32_t Width, Height;
	};

	class WindowLostFocusEvent : public WindowEvent {
	public:
		WindowLostFocusEvent() {}

		inline uint32_t GetWidth() const { return Width; }
		inline uint32_t GetHeight() const { return Height; }

		IMPLEMENT_EVENT_ENUMTYPE(EWindowEventType, WindowLostFocus);

	private:
		uint32_t Width, Height;
	};

}