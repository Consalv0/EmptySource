
#include "CoreMinimal.h"
#include "Core/Application.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/PixelMap.h"

#include "Utility/TextFormatting.h"

#include "Platform/Windows/WindowsWindow.h"

/// Remove this in the furture
#include "Platform/OpenGL/OpenGLContext.h"

#include <SDL_events.h>
#include "Platform/Windows/WindowsInput.h"
#include <SDL.h>


namespace ESource {

	void WindowsWindow::Initialize() {
		if (Context != NULL || WindowHandle != NULL) return;

		if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_JOYSTICK) != 0) {
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

		SDL_AddEventWatch(OnSDLWindowInputEvent, (void *)this);

		WindowsInput::GetInputInstance()->CheckForConnectedJoysticks();

		Context->Initialize();
	}

	WindowsWindow::WindowsWindow(const WindowProperties & Parameters) {
		Context = NULL;
		WindowHandle = NULL;
		Width = Parameters.Width;
		Height = Parameters.Height;
		Name = Parameters.Name;
		Mode = Parameters.WindowMode;
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

	void WindowsWindow::SetWindowMode(EWindowMode Mode) {
		this->Mode = Mode;
		SDL_SetWindowFullscreen((SDL_Window *)(WindowHandle), Mode);
	}

	EWindowMode WindowsWindow::GetWindowMode() const {
		return Mode;
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

	void WindowsWindow::BeginFrame() {
		CheckInputState();
	}

	void WindowsWindow::EndFrame() {
		Context->SwapBuffers();
	}

	uint64_t WindowsWindow::GetFrameCount() const {
		return Context->GetFrameCount();
	}

	void WindowsWindow::Terminate() {
		if (IsRunning()) {
#ifdef ES_DEBUG
			LOG_CORE_DEBUG(L"Window: \"{}\" closed!", GetName());
#endif // ES_DEBUG
			SDL_DestroyWindow((SDL_Window *)(WindowHandle));
			WindowHandle = NULL;
			SDL_DelEventWatch(OnSDLWindowInputEvent, (void *)this);
			Context.reset(NULL);
			// SDL_DelEventWatch(OnResizeWindow, this);
		}
	}

	void WindowsWindow::CheckInputState() {
		auto InputInstance = WindowsInput::GetInputInstance();
		for (auto & KeyStateIt : InputInstance->KeyboardInputState) {
			if (KeyStateIt.second.State & BS_Pressed) {
				KeyStateIt.second.FramePressed = Application::GetInstance()->GetWindow().GetFrameCount();
			}
			KeyStateIt.second.State &= ~(BS_Pressed | BS_Released | BS_Typed);
		}
		for (auto & MouseButtonIt : InputInstance->MouseInputState) {
			if (MouseButtonIt.second.State & BS_Pressed) {
				MouseButtonIt.second.FramePressed = Application::GetInstance()->GetWindow().GetFrameCount();
			}
			MouseButtonIt.second.State &= ~(BS_Pressed | BS_Released);
		}
		for (auto & JoystickIt : InputInstance->JoystickButtonState) {
			for (auto & JoystickButtonIt : JoystickIt.second) {
				if (JoystickButtonIt.second.State & BS_Pressed) {
					JoystickButtonIt.second.FramePressed = Application::GetInstance()->GetWindow().GetFrameCount();
				}
				JoystickButtonIt.second.State &= ~(BS_Pressed | BS_Released);
			}
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
