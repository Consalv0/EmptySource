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
		uint32_t Width;
		uint32_t Height;
		EWindowMode WindowMode;

		WindowProperties(
			const WString& Title = L"ESource",
			uint32_t Width = 1280,
			uint32_t Height = 720,
			EWindowMode WindowMode = WM_Windowed)
			: Name(Title), Width(Width), Height(Height), WindowMode(WindowMode) {
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

		//* Begin of frame functions
		virtual void BeginFrame() = 0;

		//* End of frame functions
		virtual void EndFrame() = 0;

		//* Set the window display mode
		virtual void SetWindowMode(EWindowMode Mode) = 0;

		//* Get the window display mode
		virtual EWindowMode GetWindowMode() const = 0;

		//* Get the window title name
		virtual WString GetName() const = 0;

		//* Get the width in pixels of the window
		virtual int GetWidth() const = 0;

		//* Get the height in pixels of the window
		virtual int GetHeight() const = 0;

		//* Get the size of the window in pixels
		virtual IntVector2 GetSize() const = 0;

		//* Get the size of the window in pixels
		virtual IntBox2D GetViewport() const = 0;

		//* Get the total frames renderized
		virtual uint64_t GetFrameCount() const = 0;

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
