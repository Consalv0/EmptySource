#pragma once

#include "CoreMinimal.h"
#include "Events/WindowEvent.h"
#include "Events/InputEvent.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/GraphicContext.h"

namespace ESource {

	enum EWindowMode {
		WM_Windowed = 0,
		WM_FullScreen = 1
	};

	template <typename T>
	class Bitmap;

	struct WindowProperties {
		//* Name displayed in header window
		WString Name;
		unsigned int Width;
		unsigned int Height;

		WindowProperties(
			const WString& Title = L"ESource",
			unsigned int Width = 1280,
			unsigned int Height = 720)
			: Name(Title), Width(Width), Height(Height) {
		}

	};

	//* Cointains the properties and functions of a window
	class Window {
	public:
		using WindowEventCallbackFunction = std::function<void(WindowEvent&)>;
		using InputEventCallbackFunction = std::function<void(InputEvent&)>;

		virtual ~Window() = default;

		//* Returns true if window has been created
		virtual bool IsRunning() = 0;

		//* End of frame functions
		virtual void EndFrame() = 0;

		//* Get the window title name
		virtual WString GetName() const = 0;

		//* Get the width in pixels of the window
		virtual unsigned int GetWidth() const = 0;

		//* Get the height in pixels of the window
		virtual unsigned int GetHeight() const = 0;

		//* Set callback communication with window events
		virtual void SetWindowEventCallback(const WindowEventCallbackFunction& Callback) = 0;

		//* Set callback communication with input events
		virtual void SetInputEventCallback(const InputEventCallbackFunction& Callback) = 0;

		//* Get the aspect of width divided by height in pixels of the window
		virtual float GetAspectRatio() const = 0;

		//* Get Window Pointer
		virtual void* GetHandle() const = 0;

		virtual GraphicContext * GetContext() const = 0;

		//* Creates a Window with a Name, Width and Height
		static Window * Create(const WindowProperties& Parameters = WindowProperties());
	};

}
