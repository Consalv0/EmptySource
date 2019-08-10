
#include "Engine/Log.h"
#include "Engine/Application.h"
#include "Rendering/RenderingDefinitions.h"
#include "Rendering/Bitmap.h"

#include "Utility/TextFormatting.h"

#include "Platform/Windows/WindowsWindow.h"

/// Remove this in the furture
#include "Platform/OpenGL/OpenGLContext.h"

#include <SDL.h>


namespace EmptySource {

	int OnCloseWindow(void * Data, SDL_Event * Event) {
		if (Event->type == SDL_QUIT) {
			Application::GetInstance()->ShouldClose();
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

		Context = std::unique_ptr<OpenGLContext>(new OpenGLContext((SDL_Window *)WindowHandle, 4, 6));

		SDL_AddEventWatch(OnCloseWindow, (void *)this);

		Context->Initialize();
	}

	WindowsWindow::WindowsWindow(const WindowProperties & Properties) {
		Context = NULL;
		WindowHandle = NULL;
		Width = Properties.Width;
		Height = Properties.Height;
		Name = Properties.Name;
		Mode = WM_Windowed;
		Initialize();
	}

	WindowsWindow::~WindowsWindow() {
		Terminate();
		SDL_Quit();
	}

	void WindowsWindow::Resize(const unsigned int & Wth, const unsigned int & Hht) {
		if (Width != Wth || Height != Hht) {
			Width = Wth; Height = Hht;
			SDL_SetWindowSize((SDL_Window *)(WindowHandle), Wth, Hht);
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

	Window * Window::Create(const WindowProperties& Properties) {
		return new WindowsWindow(Properties);
	}

	void WindowsWindow::SetIcon(Bitmap<UCharRGBA>* Icon) {
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

		SDL_Event Event;
		while (SDL_PollEvent(&Event)) {
		}
	}

	void WindowsWindow::Terminate() {
		if (IsRunning()) {
#ifdef ES_DEBUG
			LOG_CORE_DEBUG(L"Window: \"{}\" closed!", GetName());
#endif // ES_DEBUG
			SDL_DestroyWindow((SDL_Window *)(WindowHandle));
			WindowHandle = NULL;
			SDL_DelEventWatch(OnCloseWindow, (void *)this);
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

	unsigned int WindowsWindow::GetWidth() const {
		return Width;
	}

	unsigned int WindowsWindow::GetHeight() const {
		return Height;
	}

	void * WindowsWindow::GetHandle() const	{
		return WindowHandle;
	}

}
