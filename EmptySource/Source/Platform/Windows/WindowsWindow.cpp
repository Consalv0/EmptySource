
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/PixelMap.h"

#include "Utility/TextFormatting.h"

#include "Platform/Windows/WindowsWindow.h"

/// Remove this in the furture
#include "Platform/OpenGL/OpenGLContext.h"

#include <SDL.h>

namespace ESource {

	int OnSDLEvent(void * UserData, SDL_Event * Event) {
		WindowsWindow& Data = *(WindowsWindow*)UserData;
		static int32_t MouseButtonPressedCount[255] = { 
			(int32_t)-1, (int32_t)-1, (int32_t)-1, (int32_t)-1, (int32_t)-1, 
			(int32_t)-1, (int32_t)-1, (int32_t)-1, (int32_t)-1, (int32_t)-1
		};
		
		if (Event->type == SDL_WINDOWEVENT) {
			if (Event->window.windowID != SDL_GetWindowID((SDL_Window*)Data.GetHandle()))
				return 0;
		}

		switch (Event->type) {
		case SDL_WINDOWEVENT: {
			if (Event->window.event == SDL_WINDOWEVENT_RESIZED || Event->window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
				Data.Resize(Event->window.data1, Event->window.data2);
				break;
			}
			if (Event->window.event == SDL_WINDOWEVENT_CLOSE) {
				WindowCloseEvent WinEvent;
				Data.WindowEventCallback(WinEvent);
				break;
			}
			if (Event->window.event == SDL_WINDOWEVENT_FOCUS_GAINED) {
				WindowGainFocusEvent WinEvent;
				Data.WindowEventCallback(WinEvent);
				break;
			}
			if (Event->window.event == SDL_WINDOWEVENT_FOCUS_LOST) {
				WindowLostFocusEvent WinEvent;
				Data.WindowEventCallback(WinEvent);
				break;
			}
		}

		case SDL_KEYDOWN: {
			KeyPressedEvent InEvent(
				Event->key.keysym.scancode,
				(SDL_GetModState() & KMOD_SHIFT) != 0,
				(SDL_GetModState() & KMOD_CTRL) != 0,
				(SDL_GetModState() & KMOD_ALT) != 0,
				(SDL_GetModState() & KMOD_GUI) != 0,
				Event->key.repeat
			);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_KEYUP: {
			KeyReleasedEvent InEvent(
				Event->key.keysym.scancode,
				(SDL_GetModState() & KMOD_SHIFT) != 0,
				(SDL_GetModState() & KMOD_CTRL) != 0,
				(SDL_GetModState() & KMOD_ALT) != 0,
				(SDL_GetModState() & KMOD_GUI) != 0
			);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_TEXTINPUT: {
			KeyTypedEvent InEvent(Event->text.text);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_MOUSEBUTTONDOWN: {
			MouseButtonPressedCount[Event->button.button]++;
			MouseButtonPressedEvent InEvent(Event->button.button, Event->button.clicks == 2, MouseButtonPressedCount[Event->button.button]);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_MOUSEBUTTONUP: {
			MouseButtonPressedCount[Event->button.button] = -1;
			MouseButtonReleasedEvent InEvent(Event->button.button);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_MOUSEMOTION: {
			MouseMovedEvent InEvent((float)Event->motion.x, (float)Event->motion.y, (float)Event->motion.xrel, (float)Event->motion.yrel);
			Data.InputEventCallback(InEvent);
			break;
		}

		case SDL_MOUSEWHEEL: {
			MouseScrolledEvent InEvent(
				(float)Event->wheel.x, (float)Event->wheel.y,
				Event->wheel.direction == SDL_MOUSEWHEEL_FLIPPED
			);
			Data.InputEventCallback(InEvent);
			break;
		}
		}

		return 0;
	}

	void WindowsWindow::Initialize() {

		if (Context != NULL || WindowHandle != NULL) return;

		if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
			LOG_CORE_CRITICAL(L"Failed to initialize SDL 2.0.9: {0}\n", Text::NarrowToWide(SDL_GetError()));
			return;
		}

		if ((WindowHandle = SDL_CreateWindow(
			Text::WideToNarrow(Name).c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, Width, Height,
			SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN | Mode
		)) == NULL) {
			LOG_CORE_CRITICAL(L"Window: \"{0}\" could not be initialized: {1}", Name, Text::NarrowToWide(SDL_GetError()));
			return;
		}

		SDL_SetWindowData((SDL_Window *)WindowHandle, "WindowData", this);

		Context = std::unique_ptr<OpenGLContext>(new OpenGLContext((SDL_Window *)WindowHandle, 4, 6));

		SDL_AddEventWatch(OnSDLEvent, (void *)this);

		Context->Initialize();
	}

	WindowsWindow::WindowsWindow(const WindowProperties & Parameters) {
		Context = NULL;
		WindowHandle = NULL;
		Width = Parameters.Width;
		Height = Parameters.Height;
		Name = Parameters.Name;
		Mode = WM_Windowed;
		Initialize();
	}

	WindowsWindow::~WindowsWindow() {
		Terminate();
		SDL_Quit();
	}

	void WindowsWindow::Resize(const uint32_t & Wth, const uint32_t & Hht) {
		if (Width != Wth || Height != Hht) {
			Width = Wth; Height = Hht;
			SDL_SetWindowSize((SDL_Window *)(WindowHandle), Wth, Hht);

			WindowResizeEvent Event(Width, Height);
			WindowEventCallback(Event);
		}
	}

	void WindowsWindow::SetName(const WString & NewName) {
		Name = NewName;
		SDL_SetWindowTitle((SDL_Window *)(WindowHandle), Text::WideToNarrow(Name).c_str());
	}

	Vector2 WindowsWindow::GetMousePosition(bool Clamp) {
		int MouseX, MouseY;
		if (Clamp) {
			SDL_GetMouseState(&MouseX, &MouseY);
		}
		else {
			int WindowX, WindowY;
			SDL_GetGlobalMouseState(&MouseX, &MouseY);
			SDL_GetWindowPosition((SDL_Window *)(WindowHandle), &WindowX, &WindowY);
			MouseX -= WindowX;
			MouseY -= WindowY;
		}
		return Vector2(float(MouseX), float(MouseY));
	}

	void WindowsWindow::SetIcon(PixelMap* Icon) {
		if (Icon->GetColorFormat() != PF_RGBA8) return;
		SDL_Surface * Surface = SDL_CreateRGBSurfaceFrom(
			(void*)Icon->PointerToValue(),
			Icon->GetWidth(), Icon->GetHeight(),
			32, 4 * Icon->GetWidth(),
			0xF000, 0x0F00, 0x00F0, 0x000F
		);
		SDL_SetWindowIcon(static_cast<SDL_Window *>(WindowHandle), Surface);
		SDL_FreeSurface(Surface);
	}

	void WindowsWindow::EndFrame() {
		Context->SwapBuffers();
	}

	void WindowsWindow::Terminate() {
		if (IsRunning()) {
#ifdef ES_DEBUG
			LOG_CORE_DEBUG(L"Window: \"{}\" closed!", GetName());
#endif // ES_DEBUG
			SDL_DestroyWindow((SDL_Window *)(WindowHandle));
			WindowHandle = NULL;
			SDL_DelEventWatch(OnSDLEvent, (void *)this);
			Context.reset(NULL);
			// SDL_DelEventWatch(OnResizeWindow, this);
		}
	}

	bool WindowsWindow::IsRunning() {
		return WindowHandle != NULL && Context->IsValid();
	}

	WString WindowsWindow::GetName() const {
		return Name;
	}

	float WindowsWindow::GetAspectRatio() const {
		return (float)Width / (float)Height;
	}

	int WindowsWindow::GetWidth() const {
		return Width;
	}

	int WindowsWindow::GetHeight() const {
		return Height;
	}

	IntVector2 WindowsWindow::GetSize() const {
		return IntVector2(Width, Height);
	}

	IntBox2D WindowsWindow::GetViewport() const {
		return IntBox2D(0, 0, Width, Height);
	}

	void * WindowsWindow::GetHandle() const	{
		return WindowHandle;
	}

	GraphicContext * WindowsWindow::GetContext() const {
		return Context.get();
	}

	Window * Window::Create(const WindowProperties& Parameters) {
		return new WindowsWindow(Parameters);
	}

}
