
#include "Core/Window.h"
#include "Rendering/GraphicContext.h"
#include "Platform/DeviceFunctions.h"
#include "Events/Property.h"
#include "Math/MathUtility.h"
#include "Math/IntVector2.h"

namespace EmptySource {

	class WindowsWindow : public Window {
	public:
		typedef Window Super;

		struct {
			WString Name;
			unsigned int Width;
			unsigned int Height;
			bool VSync;
			WindowEventCallbackFunction WindowEventCallback;
			InputEventCallbackFunction InputEventCallback;
		};

	private:

		EWindowMode Mode;

		void * WindowHandle;

		std::unique_ptr<GraphicContext> Context;

		void Initialize();
		
		// void CreateWindowEvents();
		// 
		// //* Initialize inputs in this window
		// void InitializeInputs();

	public:
		
		WindowsWindow(const WindowProperties & Parameters = WindowProperties());

		virtual ~WindowsWindow();

		inline GraphicContext & GetContext() { return *Context; };

		//* Resize the size of the window
		void Resize(const unsigned int& Width, const unsigned int& Height);

		//* Rename the window title
		void SetName(const WString & NewName);

		//* Get mouse position in screen coordinates relative to the upper left position of this window
		struct Vector2 GetMousePosition(bool Clamp = false);

		// //* Get key pressed
		// bool GetKeyDown(unsigned int Key);

		//* Sets the window icon
		void SetIcon(class PixelMap * Icon);

		//* Window update events
		virtual void EndFrame() override;

		//* Terminates this window
		void Terminate();

		//* Returns true if window has been created
		virtual bool IsRunning() override;

		//* Get the window title name
		virtual WString GetName() const override;

		//* Get the aspect of width divided by height in pixels of the window
		virtual float GetAspectRatio() const override;

		//* Get the width in pixels of the window
		virtual unsigned int GetWidth() const override;

		//* Get the height in pixels of the window
		virtual unsigned int GetHeight() const override;

		//* Get the platform Window pointer
		virtual void* GetHandle() const override;

		virtual GraphicContext* GetContext() const override;

		//* Set callback communication with window events
		inline void SetWindowEventCallback(const WindowEventCallbackFunction& Callback) override { WindowEventCallback = Callback; }

		//* Set callback communication with input events
		inline void SetInputEventCallback(const InputEventCallbackFunction& Callback) override { InputEventCallback = Callback; }

	};

}